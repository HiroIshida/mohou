from rlbench.observation_config import CameraConfig, ObservationConfig


def setup_observation_config(camera_name: str, resolution: int) -> ObservationConfig:
    camera_names = {"left_shoulder", "right_shoulder", "overhead", "wrist", "front"}
    assert camera_name in camera_names

    kwargs = {}
    ignore_camera_names = camera_names.difference(camera_name)
    for ignore_name in ignore_camera_names:
        kwargs[ignore_name + "_camera"] = CameraConfig(
            rgb=False, depth=False, point_cloud=False, mask=False
        )

    kwargs[camera_name + "_camera"] = CameraConfig(
        image_size=(resolution, resolution), point_cloud=False, mask=False
    )
    return ObservationConfig(**kwargs)
