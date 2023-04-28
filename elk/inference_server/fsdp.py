import logging
import multiprocessing as std_mp
import os
import random
import socket
import uuid
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import cycle
from typing import (
    Any,
    Callable,
    Iterable,
    Type,
    cast,
    Sequence,
    TypeAlias,
    TYPE_CHECKING,
    NewType,
    Optional,
)
from uuid import UUID

import dill
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from elk.inference_server.fsdp_options import FSDPOptions
from elk.multiprocessing import A
from elk.utils import instantiate_model, pytree_map, select_usable_devices


class SingletonSentinel:
    ...


@dataclass(kw_only=True)
class TaskMessage:
    id: UUID
    data: Dataset
    closure: Any  # dill closure


@dataclass(kw_only=True)
class ResultMessage:
    id: UUID
    data: Any


if TYPE_CHECKING:
    ResultQueue = mp.Queue[ResultMessage | Type[SingletonSentinel]]
    TaskQueue = mp.Queue[TaskMessage | Type[SingletonSentinel]]

else:
    ResultQueue = mp.Queue
    TaskQueue = mp.Queue

QueueID = NewType("QueueID", str)


def get_queue_id(uuid: UUID, rank: int) -> QueueID:
    return QueueID(f"{str(uuid)}_{rank}")


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
        num_workers: int = 1,
    ):
        self.model_str = model_str
        assert num_workers > 0
        self.num_workers = num_workers
        self.fsdp = fsdp
        self._current_id = 0
        self._process_ctx: mp.ProcessContext | None = None
        # UUID to be thread-safe
        self._manager = mp.Manager()
        self._task_queues: list[TaskQueue] = [  # type: ignore
            self._manager.Queue() for _ in range(self.num_workers)
        ]
        # QueueId to make it thread-safe
        self._result_queues: dict[QueueID, ResultQueue] = self._manager.dict()  # type: ignore
        model = instantiate_model(model_str, torch_dtype="auto")
        model.share_memory()
        self._model = model
        """Spin up the workers."""
        if self._process_ctx is not None:
            raise RuntimeError("The server is already running")

        # Load the model on the main process, then zero-copy share it with the workers.
        # This ensures that we don't copy the model num_workers times on the CPU and
        # run out of RAM for large models
        print("Loading model...")
        model = self._model
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        fdsp_min_mem = model_size / self.num_workers if self.fsdp.fsdp_enabled else None

        min_gpu_mem = (
            min_gpu_mem
            if min_gpu_mem is not None
            else fdsp_min_mem
            if fdsp_min_mem is not None
            else model_size
        )

        # Determine which GPUs we can use
        devices = select_usable_devices(self.num_workers, min_memory=min_gpu_mem)
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
        self._process_ctx = mp.spawn(
            _worker,
            args=(
                devices,
                model,
                self._task_queues,
                self._result_queues,
                cpu_offload,
                fsdp_port,
                wrap_policy,
            ),
            join=False,
            nprocs=self.num_workers,
        )

    def create_result_queues(self, _uuid: UUID, num_workers: int) -> list[QueueID]:
        """Create a queue for the given task ID."""
        queue_ids = []
        for i in range(num_workers):
            queue_id = get_queue_id(_uuid, i)
            new_queue = self._manager.Queue()
            self._result_queues[queue_id] = new_queue  # type: ignore
            queue_ids.append(queue_id)
        return queue_ids

    @property
    def running(self) -> bool:
        """Whether the server is running."""
        return self._process_ctx is not None

    def shutdown(self) -> bool:
        """Shut down all the workers, returning `True` if successful."""
        if self._process_ctx is None:
            raise RuntimeError("Can't shut down a server that isn't running")

        # Let the workers know that they should shut down
        for q in self._task_queues:
            try:
                q.put_nowait(SingletonSentinel)
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
        self,
        closure: Callable[[ModelOutput], A],
        dataset: Dataset,
    ) -> list[A]:
        """Run inference on the given inputs, running a closure on the outputs.
        Note that the order of the outputs is not guaranteed to match
        """
        return list(self.imap(closure, dataset))

    def one(
        self,
        dataset: Dataset,
    ) -> ModelOutput:
        """Run inference on the given input, running a closure on the outputs."""
        return self.map(lambda x: x, dataset)[0]

    def imap(
        self,
        closure: Callable[[ModelOutput], A],
        dataset: Dataset,
    ) -> Iterable[A]:
        """Run inference on the given inputs, running a closure on the outputs."""
        if self._process_ctx is None:
            raise RuntimeError("Can't run inference on a server that isn't running")

        # Pickle the closure and send it to the workers
        closure_pkl = dill.dumps(closure)
        shards = [dataset.shard(self.num_workers, i) for i in range(self.num_workers)]
        # shuffle the shards so that we distribute the load evenly
        shuffled_shards = random.sample(shards, len(shards))
        # create a uuid to track what messages belong to what imap on different threads
        _id: UUID = uuid.uuid4()
        queue_ids = self.create_result_queues(_uuid=_id, num_workers=self.num_workers)
        for task_q, shard in zip(
            self._task_queues,
            shuffled_shards,
        ):
            message = TaskMessage(id=_id, closure=closure_pkl, data=shard)
            task_q.put(message)

        yield from round_robin(
            sentinel=SingletonSentinel,
            queue_ids=queue_ids,
            result_queue_dict=self._result_queues,
        )


@torch.inference_mode()
def _worker(
    rank: int,
    devices: list[str],
    model: PreTrainedModel,
    qs: list[TaskQueue],
    out_qs: dict[QueueID, ResultQueue],
    cpu_offload: bool = False,
    fsdp_port: int | None = None,
    wrap_policy: partial[bool] | None = None,
):
    """Worker process that maintains a copy of the model on a dedicated GPU."""
    print("Starting worker!")
    # Prevent duplicate logging messages
    if rank != 0:
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

    closure: Callable[[ModelOutput], Any]
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
                forward_prefetch=True,
            )
            model = cast(PreTrainedModel, wrapped)
            # This is dumb, but we need to run a forward pass to initialize the
            # FSDP. Otherwise, the first forward pass on all workers will not run
            # if one of the workers doesn't receive a dataset to run on
            # This can happen if the first dataset len is lesser than the number of
            # workers
            model(input_ids=torch.Tensor([0]).long().to(device))
            print(f"FSDP running on rank {rank} with {device}")
        else:
            model.to(device)

        # Breaks when x is None, the sentinel value indicating we should shut down
        in_queue = qs[rank]

        while msg := in_queue.get():
            if isinstance(msg, SingletonSentinel):
                # Someone called shutdown() on the server
                print("Shutting down worker")
                return
            print("Got msg")
            # Someone called map() giving us a new closure and dataset to use
            assert isinstance(msg, TaskMessage)
            closure_pkl = msg.closure
            dataset = msg.data
            closure = dill.loads(closure_pkl)
            queue_id = get_queue_id(msg.id, rank=rank)
            # We need to send the results back to the correct queue
            out_queue = out_qs[queue_id]
            for record in dataset:
                assert isinstance(record, dict)
                try:
                    inputs_cuda = pytree_map(
                        lambda t: t.to(device).unsqueeze(0), record
                    )
                except Exception as e:
                    print(f"Failed to move inputs to cuda: {e}")
                    raise e

                # We always want to return the hidden states
                try:
                    outputs = model(**inputs_cuda, output_hidden_states=True)
                except Exception as e:
                    print(f"forward failed {e}")
                    raise e

                outputs_cls = type(outputs)
                outputs_dict = pytree_map(lambda x: x.cpu().share_memory_(), outputs)
                outputs = outputs_cls(**outputs_dict)
                # apply the closure
                output_applied = closure(outputs)

                # Send the outputs back to the main process
                out_queue.put(ResultMessage(data=output_applied, id=msg.id))

            # Indicate we're done with this dataset
            out_queue.put_nowait(SingletonSentinel)

        # Clean up the FSDP process group
        if fsdp_port is not None:
            dist.destroy_process_group()
    except Exception as e:
        print(f"Worker failed with {e}, Type: {type(e)}")
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
    sentinel: Type[SingletonSentinel],
    queue_ids: list[QueueID],
    result_queue_dict: dict[QueueID, ResultQueue],
) -> Iterable[Any]:
    """Yield items from the given queues in round-robin order."""
    result_queue = [result_queue_dict[queue_id] for queue_id in queue_ids]
    remaining_queues = len(result_queue)

    for idx, q in cycle(enumerate(result_queue)):
        if remaining_queues == 0:
            print("breaking the generator")
            # We've exhausted all queues. Delete the queues and break
            for queue_id in queue_ids:
                del result_queue_dict[queue_id]
            break
        try:
            item = q.get(timeout=0.01)
        except std_mp.queues.Empty:  # type: ignore[attr-defined]
            continue
        else:
            if item == sentinel:
                print("Got sentinel")
                remaining_queues -= 1
            else:
                yield cast(ResultMessage, item).data
