import logging
import multiprocessing as std_mp
import os
import socket
import traceback
import uuid
import warnings
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    NewType,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    cast,
)
from uuid import UUID

import dill
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from elk.inference_server.fsdp_options import FSDPOptions
from elk.utils import instantiate_model, pytree_map, select_usable_devices
from elk.utils.typing import A

sentinel: Literal["sentinel"] = "sentinel"
SingletonSentinel: TypeAlias = Literal["sentinel"]

ResultQueueID = NewType("QueueID", str)


@dataclass(kw_only=True)
class TaskMessage:
    id: ResultQueueID
    data: Sequence[dict]
    func_dill: bytes | None

    def func(self) -> Callable[[ModelOutput], Any] | None:
        return dill.loads(self.func_dill) if self.func_dill else None


@dataclass(kw_only=True)
class ResultMessage:
    id: ResultQueueID
    # Either ModelOutput or something applied by the func to the output
    data: Any
    exception: Exception | None = None


if TYPE_CHECKING:
    ResultQueue = mp.Queue[ResultMessage | SingletonSentinel]
    TaskQueue = mp.Queue[TaskMessage | SingletonSentinel]

else:
    ResultQueue = mp.Queue
    TaskQueue = mp.Queue


def get_queue_id(_uuid: UUID) -> ResultQueueID:
    """
    _uuid: UUID of the task. E.g. for a map operation, this is the UUID of the map
    rank: rank of the worker. Each worker will output to a different queue
    """
    return ResultQueueID(str(_uuid))


def shard_seq(seq: Sequence[A], num_shards: int) -> Sequence[Sequence[A]]:
    """Shard a sequence into `num_shards` chunks."""
    return [seq[i::num_shards] for i in range(num_shards)]


class InferenceServer:
    """High-level interface for running inference on a model on multiple GPUs.

    This is basically a glorified `multiprocessing.Pool`. The only difference is that
    each worker maintains a copy of the model on a dedicated GPU.
    """

    def __init__(
        self,
        model_str: str,
        fsdp: FSDPOptions,
        min_gpu_mem: Optional[float | int],
        num_workers: int,
        mp_sharing_strategy: Literal["file_system", "file_descriptor"],
    ):
        self.model_str = model_str
        self.fsdp = fsdp
        self._current_id = 0
        self._manager = mp.Manager()
        # Single task_queue, so that the tasks are distributed evenly
        self._task_queue: TaskQueue = self._manager.Queue()  # type: ignore
        # Multiple result_queues, for each run of map / infer
        # so that its thread-safe
        self._result_queues: dict[
            ResultQueueID, ResultQueue
        ] = self._manager.dict()  # type: ignore
        model = instantiate_model(model_str, torch_dtype="auto")
        model.share_memory()
        self._model = model
        """Spin up the workers."""
        # Load the model on the main process, then zero-copy share it with the workers.
        # This ensures that we don't copy the model num_workers times on the CPU and
        # run out of RAM for large models
        print("Loading model...")
        model = self._model
        # to fix: this model_size doesn't seem to work
        # model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # fdsp_min_mem = (
        #     model_size / num_workers
        #     if self.fsdp.fsdp_enabled and num_workers > 0
        #     else None
        # )
        #

        # min_gpu_mem = (
        #     min_gpu_mem
        #     if min_gpu_mem is not None
        #     else fdsp_min_mem
        #     if fdsp_min_mem is not None
        #     else model_size
        # )
        # Divide the min_gpu_mem by the number of workers for fsdp
        maybe_divided_min_gpu_mem: float | int | None = (
            (min_gpu_mem * 1.1) / num_workers
            if self.fsdp.fsdp_enabled and min_gpu_mem is not None
            else min_gpu_mem
        )

        if maybe_divided_min_gpu_mem is not None:
            print(
                f"Requiring at least {maybe_divided_min_gpu_mem / 1e9}"
                f" gb of GPU memory per worker for fsdp"
            )

        # Determine which GPUs we can use
        devices = select_usable_devices(
            num_workers, min_memory=maybe_divided_min_gpu_mem
        )
        print(f"Using devices {devices}")
        self.devices = devices
        self.num_workers = len(devices)  # This may have been -1 before
        fsdp_port, wrap_policy = None, None
        if self.fsdp.fsdp_enabled:
            fsdp_port = find_available_port()
            msg = f"Fully Sharded Data Parallel running on port {fsdp_port}"

            layer_cls = get_transformer_layer_cls(model)
            if layer_cls is not None:
                msg += f" with '{layer_cls.__name__}' wrapping policy"
                wrap_policy = partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
                )

            print(msg)
        cpu_offload: bool = fsdp.cpu_offload if fsdp else False
        mp.set_sharing_strategy(mp_sharing_strategy)
        self._process_ctx: mp.ProcessContext = mp.spawn(  # type: ignore
            _worker,
            args=(
                devices,
                model,
                self._task_queue,
                self._result_queues,
                cpu_offload,
                fsdp_port,
                wrap_policy,
            ),
            join=False,
            nprocs=self.num_workers,
        )

    def create_result_queue(self, _uuid: UUID) -> ResultQueueID:
        """Create a queue for the given task ID."""
        queue_id = ResultQueueID(str(_uuid))
        new_queue = self._manager.Queue()
        self._result_queues[queue_id] = new_queue  # type: ignore
        return queue_id

    def shutdown(self) -> bool:
        """Shut down all the workers, returning `True` if successful."""
        # Let the workers know that they should shut down
        for _ in range(self.num_workers):
            try:
                self._task_queue.put_nowait(sentinel)
            except std_mp.queues.Empty:  # type: ignore[attr-defined]
                pass

        self._manager.shutdown()
        return self._process_ctx.join()

    # Support use as a context manager, just like mp.Pool
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def map(
        self, keywords: list[dict[str, Any]], func: Callable[[ModelOutput], A]
    ) -> list[A]:
        """Run inference on the given inputs, running a func on the outputs.
        Note that the order of the outputs is not guaranteed to match
        """
        return list(self.imap(keywords=keywords, func=func))

    def imap(
        self,
        keywords: list[dict[str, Any]],
        func: Callable[[ModelOutput], A],
    ) -> Iterable[A]:
        """Run inference on the given inputs, running a func on the outputs."""

        # Pickle the func and send it to the workers
        func_dill = dill.dumps(func)
        shards = shard_seq(keywords, self.num_workers)
        # create a uuid to track what messages belong to what imap on different threads
        _id: UUID = uuid.uuid4()
        queue_id: ResultQueueID = self.create_result_queue(_uuid=_id)
        for shard in shards:
            # move the kwargs to the cpu device and share memory
            new_data = [
                share_dict_of_tensors_with_processes(kwargs) for kwargs in shard
            ]
            message = TaskMessage(id=queue_id, func_dill=func_dill, data=new_data)
            self._task_queue.put(message)

        yield from round_robin(
            queue_id=queue_id,
            num_to_wait=len(shards),
            result_queue_dict=self._result_queues,
        )

    def infer(self, **kwargs) -> ModelOutput:
        """Run inference on the given input. These are passed directly to the model."""
        # Optimized version of map for one input. No need to create so many queues.
        # Pick the first available worker
        _id: UUID = uuid.uuid4()
        queue_id = get_queue_id(_id)
        result_queue = self._manager.Queue()
        self._result_queues[queue_id] = result_queue  # type: ignore
        # move the kwargs to the cpu device and share memory
        shared_kwargs = share_dict_of_tensors_with_processes(kwargs)
        # Send the task to the worker
        message = TaskMessage(id=queue_id, func_dill=None, data=[shared_kwargs])
        self._task_queue.put(message)

        # Wait for the result
        result_msg: ResultMessage = result_queue.get()
        if result_msg.exception is not None:
            raise result_msg.exception
        else:
            output = result_msg.data

        # Clean up the result queue
        del self._result_queues[queue_id]

        return output


def identity(x):
    return x


def share_dict_of_tensors_with_processes(_dict: dict[str, torch.Tensor]):
    """Share a dictionary of tensors with all the processes."""
    return pytree_map(lambda x: x.cpu().share_memory_(), _dict)


@torch.inference_mode()
def _worker(
    rank: int,
    devices: list[str],
    model: PreTrainedModel,
    task_queue: TaskQueue,
    out_qs: dict[ResultQueueID, ResultQueue],
    cpu_offload: bool = False,
    fsdp_port: int | None = None,
    wrap_policy: partial[bool] | None = None,
):
    """Worker process that maintains a copy of the model on a dedicated GPU."""
    # Prevent duplicate logging messages
    if rank != 0:
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

    func: Callable[[ModelOutput], Any]
    device = devices[rank]

    try:
        # Fully Sharded Data Parallel for large models
        if fsdp_port is not None:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(fsdp_port)
            dist.init_process_group("nccl", rank=rank, world_size=len(devices))
            torch.cuda.set_device(device)

            wrapped = FSDP(
                model,
                auto_wrap_policy=wrap_policy,
                cpu_offload=CPUOffload(offload_params=cpu_offload),
                device_id=torch.device(device),
                # Since we are inference only, we don't need to sync the nn.modules
                sync_module_states=False,
                forward_prefetch=True,
            )
            # torch.compile() needs to be called after wrapping the model with FSDP
            wrapped_compiled = torch.compile(wrapped)
            model = cast(PreTrainedModel, wrapped_compiled)
            # This is dumb, but we need to run a forward pass to initialize the
            # FSDP. Otherwise, the first forward pass on all workers will not run
            # if one of the workers doesn't something to run on
            # This can happen if the first batch is lesser than the number of
            # workers
            model(input_ids=torch.Tensor([[0]]).long().to(device))
            print(
                f"FSDP running on rank {rank} with {device} and cpu_offload {cpu_offload}"
            )
        else:
            model.to(device)

        # Breaks when x is None, the sentinel value indicating we should shut down

        while msg := task_queue.get():
            if msg == sentinel:
                # Someone called shutdown() on the server
                print("Shutting down worker")
                return
            # Someone called map() giving us a new func and dataset to use
            assert isinstance(msg, TaskMessage)
            data = msg.data
            func = msg.func()  # type: ignore
            queue_id = msg.id
            # We need to send the results back to the correct queue
            result_queue = out_qs[queue_id]
            try:
                for record in data:
                    assert isinstance(record, dict)
                    inputs_cuda = pytree_map(lambda t: t.to(device), record)
                    # testing this to make sure it's really inference mode
                    # despite the decorator
                    with torch.inference_mode():
                        # We always want to return the hidden states
                        outputs = model(**inputs_cuda, output_hidden_states=True)

                    outputs_cls = type(outputs)
                    outputs_dict = pytree_map(
                        lambda x: x.cpu().share_memory_(), outputs
                    )
                    outputs = outputs_cls(**outputs_dict)
                    # apply the func
                    output_applied = func(outputs) if func is not None else outputs

                    # Send the outputs back to the main process
                    result_queue.put(ResultMessage(data=output_applied, id=msg.id))
            except Exception as e:
                # Send the exception back to the main process
                result_queue.put(ResultMessage(exception=e, id=msg.id, data=None))

            # Indicate we're done with this dataset
            result_queue.put_nowait(sentinel)

        # Clean up the FSDP process group
        if fsdp_port is not None:
            dist.destroy_process_group()
    except Exception as e:
        print(f"Worker failed with {e}, Type: {type(e)}")
        print(traceback.format_exc())
        raise e


def get_transformer_layer_cls(model: torch.nn.Module) -> Type[torch.nn.Module] | None:
    """Get the class of the transformer layer used by the given model."""
    total_params = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > total_params / 2:
                return type(module[0])

    return None


def get_socket_with_port() -> socket.socket:
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError:
            s.close()

    raise RuntimeError("Failed to create a socket")


def find_available_port() -> int:
    s = get_socket_with_port()
    _, port = s.getsockname()
    s.close()

    return port


def round_robin(
    queue_id: ResultQueueID,
    num_to_wait: int,
    result_queue_dict: dict[ResultQueueID, ResultQueue],
) -> Iterable[Any]:
    result_queue = result_queue_dict[queue_id]
    remaining_workers = num_to_wait

    while remaining_workers > 0:
        try:
            item: ResultMessage | SingletonSentinel = result_queue.get(timeout=0.01)
        except std_mp.queues.Empty:  # type: ignore[attr-defined]
            continue
        else:
            if item == sentinel:
                remaining_workers -= 1
            elif item.exception is not None:  # type: ignore
                # We got an exception from the worker. Raise it here
                raise item.exception  # type: ignore
            else:
                yield cast(ResultMessage, item).data

    # Clean up the result queue
    del result_queue_dict[queue_id]
