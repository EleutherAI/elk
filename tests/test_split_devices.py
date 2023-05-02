from elk.utils.multi_gpu import ModelDevices, split_devices_into_model_devices


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
