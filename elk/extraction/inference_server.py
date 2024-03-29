import inspect
import logging
import multiprocessing as std_mp
import os
import socket
import warnings
from dataclasses import dataclass
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
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from elk.utils import instantiate_model, pytree_map, select_usable_devices


@dataclass(frozen=True)
class _Sentinel:
    """Sentinel value used to indicate that a worker is done."""


SENTINEL = _Sentinel()


@dataclass
class InferenceServer:
    """High-level interface for running inference on a model on multiple GPUs.

    This is basically a glorified `multiprocessing.Pool`. The only difference is that
    each worker maintains a copy of the model on a dedicated GPU.
    """

    model_str: str
    num_workers: int = -1
    cpu_offload: bool = False
    fsdp: bool = False

    def __post_init__(self):
        self._current_id = 0
        self._process_ctx: mp.ProcessContext | None = None

        self._result_queues = []
        self._task_queues = []

    @property
    def running(self) -> bool:
        """Whether the server is running."""
        return self._process_ctx is not None

    def start(self) -> None:
        """Spin up the workers."""
        if self._process_ctx is not None:
            raise RuntimeError("The server is already running")

        # Load the model on the main process, then zero-copy share it with the workers.
        # This ensures that we don't copy the model num_workers times on the CPU and
        # run out of RAM for large models
        print("Loading model...")
        model = instantiate_model(self.model_str, torch_dtype="auto")
        model.share_memory()
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Determine which GPUs we can use
        devices = select_usable_devices(
            self.num_workers, min_memory=model_size if not self.fsdp else None
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
            _worker_wrapper,
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
                q.put_nowait(None)
            except std_mp.queues.Empty:  # type: ignore[attr-defined]
                pass

        self._manager.shutdown()
        return self._process_ctx.join()

    # Support use as a context manager, just like mp.Pool
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def map_forward(
        self,
        dataset: Dataset,
        model_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> list:
        """Maps the model's `forward` method over the given dataset, without
        running a closure on the outputs."""
        return self.map(
            lambda x: x, dataset, model_kwargs=model_kwargs, use_tqdm=use_tqdm
        )

    def imap_forward(
        self,
        dataset: Dataset,
        model_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> Iterable:
        """Maps the model's `forward` method over the given dataset, without
        running a closure on the outputs."""
        yield from self.imap(
            lambda x: x, dataset, model_kwargs=model_kwargs, use_tqdm=use_tqdm
        )

    def map(
        self,
        closure: Callable[[ModelOutput], Any],
        dataset: Dataset,
        model_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> list:
        """Run inference on the given inputs, running a closure on the outputs.
        Dataset must contain an `input_ids` column, and optionally other arguments
        that the model expects."""
        # add id column to dataset if not present to keep track of order
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", range(len(dataset)))  # type: ignore
        ids = dataset["id"]
        output_tuples = list(self.imap(closure, dataset, model_kwargs, use_tqdm))
        outputs = dict(output_tuples)
        return [outputs[id] for id in ids]

    def imap(
        self,
        closure: Callable[[ModelOutput], Any] | None,
        dataset: Dataset,
        model_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> Iterable:
        """Run inference on the given inputs, running a closure on the outputs.
        Dataset must contain an `input_ids` column, and optionally other arguments
        that the model expects. `dataset` is also required to have an `id` column,
        because the outputs are not guaranteed to be returned in the same order as
        the inputs.

        yields: (id, outputs)"""
        if self._process_ctx is None:
            raise RuntimeError("Can't run inference on a server that isn't running")

        assert "id" in dataset.column_names, "Dataset must contain an 'id' column"
        if len(dataset) % self.num_workers != 0:
            # server requires that the dataset's length is a multiple of the world size
            assert self.num_workers != -1

            # duplicate some rows
            num_rows = len(dataset)
            num_needed = self.num_workers - (num_rows % self.num_workers)
            dummy = dataset[0]
            dummy_id = dummy["id"]
            for _ in range(num_needed):
                dataset = dataset.add_item(dummy)  # type: ignore
        else:
            dummy_id = -1

        # We need PyTorch tensors
        dataset = dataset.with_format("torch")

        # Pickle the closure and send it to the workers
        closure_pkl = dill.dumps(closure)
        model_kwargs_pkl = dill.dumps(model_kwargs or {})
        shards = [dataset.shard(self.num_workers, i) for i in range(self.num_workers)]
        for q, shard in zip(self._task_queues, shards):
            q.put((closure_pkl, model_kwargs_pkl, shard))

        generator = round_robin(self._result_queues)  # type: ignore[arg-type]
        seen_ids = set()
        for out in tqdm(generator, total=len(dataset), disable=not use_tqdm):
            if out[0] == dummy_id:
                if dummy_id in seen_ids:
                    continue  # ignore any extra dummy rows
            elif out[0] in seen_ids:
                raise RuntimeError(
                    "Round robin yielded duplicate items. "
                    "This may be due to multiprocessing queues returning "
                    "items repeatedly."
                )
            seen_ids.add(out[0])
            yield out


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
    _, port, *_ = s.getsockname()
    s.close()

    return port


def round_robin(queues: list[mp.Queue]) -> Iterable[Any]:
    """Yield items from the given queues in round-robin order."""
    exhausted = set()

    for idx, q in cycle(enumerate(queues)):
        if len(exhausted) == len(queues):
            break
        if idx in exhausted:
            continue

        try:
            item = q.get(timeout=0.01)
        except std_mp.queues.Empty:  # type: ignore[attr-defined]
            pass
        else:
            if item == SENTINEL:
                exhausted.add(idx)
            else:
                yield item


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
    # Prevent duplicate logging messages
    if rank != 0:
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

    closure: Callable[[ModelOutput], Any] | None = None
    dataset: Dataset | None = None
    device = devices[rank]

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
        model_forward = model.module.forward  # type: ignore[union-attr]
    else:
        model.to(device)  # type: ignore[union-attr]
        model_forward = model.forward

    param_names = set(inspect.signature(model_forward).parameters.keys())

    # Breaks when x is the sentinel value indicating we should shut down
    in_queue = qs[rank]
    out_queue = out_qs[rank]

    while msg := in_queue.get():
        # Someone called map() giving us a new closure and dataset to use
        assert isinstance(msg, tuple) and len(msg) == 3
        closure_pkl, model_kwargs_pkl, dataset = msg
        closure = dill.loads(closure_pkl)
        model_kwargs = dill.loads(model_kwargs_pkl)

        assert dataset is not None
        for record in dataset:
            assert isinstance(record, dict)
            id = record["id"].item()
            assert "input_ids" in record, "Dataset must contain an 'input_ids' column"
            # Only pass the arguments that the model expects
            input_record = {k: v for k, v in record.items() if k in param_names}

            def maybe_unsqueeze(v):
                return v.unsqueeze(0) if v.ndim == 1 else v

            inputs_cuda = pytree_map(
                lambda v: maybe_unsqueeze(v.to(device)), input_record
            )
            outputs = model(**inputs_cuda, **model_kwargs)

            if callable(closure):
                outputs = closure(outputs, **record)
            if outputs is not None:
                # Move the outputs back to the CPU
                outputs = pytree_map(lambda x: x.cpu().share_memory_(), outputs)

            # Send the outputs back to the main process
            out_queue.put((id, outputs))

        # Indicate we're done with this dataset
        out_queue.put(SENTINEL)

    # Clean up the FSDP process group
    if fsdp_port is not None:
        dist.destroy_process_group()


def _worker_wrapper(rank: int, *args):
    try:
        return _worker(rank, *args)
    except Exception as e:
        print(f"Exception in worker {rank}: {e}")
        raise e
