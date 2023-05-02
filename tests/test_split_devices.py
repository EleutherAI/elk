from elk.extraction.llama.device_configs import ModelDevices


def split_devices_into_model_devices(
    devices: list[str], gpus_per_model: int, models_to_create: int
) -> list[ModelDevices]:
    assert len(devices) >= gpus_per_model * models_to_create
    configs = []
    while len(configs) < models_to_create:
        first_device = devices.pop(0)
        other_devices = devices[: gpus_per_model - 1]
        devices = devices[gpus_per_model - 1 :]
        configs.append(ModelDevices(first_device, other_devices))
    return configs


def test_split_2_devices_1_gpu_per_model():
    devices = ["a", "b"]
    gpus_per_model = 1
    models_to_create = 2
    assert split_devices_into_model_devices(
        devices=devices,
        gpus_per_model=gpus_per_model,
        models_to_create=models_to_create,
    ) == [ModelDevices("a", []), ModelDevices("b", [])]


def test_split_4_devices_2_gpus_per_model():
    devices = ["a", "b", "c", "d"]
    gpus_per_model = 2
    models_to_create = 2
    assert split_devices_into_model_devices(
        devices=devices,
        gpus_per_model=gpus_per_model,
        models_to_create=models_to_create,
    ) == [ModelDevices("a", ["b"]), ModelDevices("c", ["d"])]


def test_split_7_devices_3_gpus_per_model():
    devices = ["a", "b", "c", "d", "e", "f", "g"]
    gpus_per_model = 3
    models_to_create = 2
    assert split_devices_into_model_devices(
        devices=devices,
        gpus_per_model=gpus_per_model,
        models_to_create=models_to_create,
    ) == [ModelDevices("a", ["b", "c"]), ModelDevices("d", ["e", "f"])]
