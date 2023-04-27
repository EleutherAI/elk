import logging
import multiprocessing as std_mp
import os
import socket
import warnings
from functools import partial
from itertools import cycle
from typing import Any, Callable, Iterable, Type, cast

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

from ..multiprocessing import A
from ..utils import instantiate_model, pytree_map, select_usable_devices


class SingletonSentinel:
    ...


class InferenceServer:
    """High-level interface for running inference on a model on multiple GPUs.

    This is basically a glorified `multiprocessing.Pool`. The only difference is that
    each worker maintains a copy of the model on a dedicated GPU.
    """

    def __init__(
        self,
        model_str: str,
        num_workers: int = 1,
        cpu_offload: bool = False,
        fsdp: bool = False,
    ):
        self.model_str = model_str
        assert num_workers > 0
        self.num_workers = num_workers
        self.cpu_offload = cpu_offload
        self.fsdp = fsdp
        self._current_id = 0
        self._process_ctx: mp.ProcessContext | None = None

        self._result_queues = []
        self._task_queues = []
        model = instantiate_model(model_str, torch_dtype="auto")
        model.share_memory()
        self._model = model
        self._start()

    @property
    def running(self) -> bool:
        """Whether the server is running."""
        return self._process_ctx is not None

    def _start(self) -> None:
        """Spin up the workers."""
        if self._process_ctx is not None:
            raise RuntimeError("The server is already running")

        # Load the model on the main process, then zero-copy share it with the workers.
        # This ensures that we don't copy the model num_workers times on the CPU and
        # run out of RAM for large models
        print("Loading model...")
        model = self._model
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())

        fdsp_min_mem = model_size / self.num_workers if self.fsdp else None

        # Determine which GPUs we can use
        devices = select_usable_devices(
            self.num_workers,
            min_memory=model_size if fdsp_min_mem is None else fdsp_min_mem,
        )
        self.num_workers = len(devices)  # This may have been -1 before

        fsdp_port, wrap_policy = None, None
        if self.fsdp:
            fsdp_port = find_available_port()
            msg = f"Fully Sharded Data Parallel running on port {fsdp_port}"

            layer_cls = get_transformer_layer_cls(model)
            if layer_cls is not None:
                msg += f" with '{layer_cls.__name__}' wrapping policy"
                wrap_policy = partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
                )

            print(msg)

        self._manager = mp.Manager()
        self._result_queues = [self._manager.Queue() for _ in range(self.num_workers)]
        self._task_queues = [self._manager.Queue() for _ in range(self.num_workers)]
        self._process_ctx = mp.spawn(
            _worker,
            args=(
                devices,
                model,
                self._task_queues,
                self._result_queues,
                self.cpu_offload,
                fsdp_port,
                wrap_policy,
            ),
            join=False,
            nprocs=self.num_workers,
        )

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

    def map_for_non_fsdp(
        self,
        closure: Callable[[ModelOutput], A],
        dataset: Dataset,
    ) -> list[A]:
        """Run inference on the given inputs, running a closure on the outputs.
        Note that the order of the outputs is not guaranteed to match
        """
        return list(self.imap_for_non_fsdp(closure, dataset))

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
        result_queues = []
        for q, result_queue in zip(self._task_queues, self._result_queues):
            # Put the same dataset on each queue, so that each worker gets the same
            # inputs
            q.put((closure_pkl, dataset))
            result_queues.append(result_queue)

        yield from round_robin(result_queues, sentinel=SingletonSentinel)  # type: ignore

    def imap_for_non_fsdp(  # todo: delete
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
        result_queues = []
        for q, shard, result_queue in zip(
            self._task_queues, shards, self._result_queues
        ):
            q.put((closure_pkl, shard))
            result_queues.append(result_queue)

        yield from round_robin(result_queues, sentinel=SingletonSentinel)  # type: ignore


@torch.inference_mode()
def _worker(
    rank: int,
    devices: list[str],
    model: PreTrainedModel,
    qs: list[mp.Queue],
    out_qs: list[mp.Queue],
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
    dataset: Dataset | None = None
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
            model(input_ids=torch.Tensor([0]).long().to(device))
            print(f"FSDP running on rank {rank} with {device}")
        else:
            model.to(device)

        # Breaks when x is None, the sentinel value indicating we should shut down
        in_queue = qs[rank]
        out_queue = out_qs[rank]

        while msg := in_queue.get():
            if isinstance(msg, SingletonSentinel):
                # Someone called shutdown() on the server
                print("Shutting down worker")
                return
            print("Got msg")
            # Someone called map() giving us a new closure and dataset to use
            assert isinstance(msg, tuple) and len(msg) == 2, "Expected a tuple"
            closure_pkl, dataset = msg
            closure = dill.loads(closure_pkl)
            print(f"Loaded closure and dataset: {dataset}")

            assert dataset is not None, "Dataset should not be None"
            if len(dataset) == 0:
                # this is dumb, but we need to run the model once to make FSDP happy
                print("Empty dataset, skipping")
                model()
            for record in dataset:
                assert isinstance(record, dict)
                try:
                    inputs_cuda = pytree_map(
                        lambda t: t.to(device).unsqueeze(0), record
                    )
                except Exception as e:
                    print(f"Failed to move inputs to cuda: {e}")
                    raise e
                print("Got cuda inputs")

                # We always want to return the hidden states
                try:
                    print(f"Running forward with {inputs_cuda} on device {device}")
                    outputs = model(**inputs_cuda, output_hidden_states=True)
                    print("Done running forward")
                except Exception as e:
                    print(f"forward failed {e}")
                    raise e
                print("Ran forward successfully")

                outputs_cls = type(outputs)
                outputs_dict = pytree_map(lambda x: x.cpu().share_memory_(), outputs)
                outputs = outputs_cls(**outputs_dict)
                # apply the closure
                output_applied = closure(outputs)

                # Send the outputs back to the main process
                out_queue.put(output_applied)

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


def round_robin(queues: list[mp.Queue], sentinel: SingletonSentinel) -> Iterable[Any]:
    """Yield items from the given queues in round-robin order."""
    remaining_queues = len(queues)

    count = 0
    for idx, q in cycle(enumerate(queues)):
        count += 1
        if count % 1000 == 0:
            print(f"Round robin: {count}")
        if remaining_queues == 0:
            print("breaking the generator")
            break

        try:
            item = q.get(timeout=5)
        except std_mp.queues.Empty:  # type: ignore[attr-defined]
            continue
        else:
            if item == sentinel:
                print("Got sentinel")
                remaining_queues -= 1
            else:
                yield item
