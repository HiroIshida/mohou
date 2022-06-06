from typing import Callable, Optional

import albumentations as al
import numpy as np

RandImageFunc = Optional[Callable[[np.ndarray], np.ndarray]]
_f_randomize_rgb_image: RandImageFunc = None
_f_randomize_gray_image: RandImageFunc = None
_f_randomize_depth_image: RandImageFunc = None


def configure_rgb_image_randomizer(rgb_shift_limit=40):
    def randomize_rgb_image(image_arr: np.ndarray):
        aug_guass = al.GaussNoise(p=1)
        sl = rgb_shift_limit
        aug_rgbshit = al.RGBShift(r_shift_limit=sl, g_shift_limit=sl, b_shift_limit=sl)
        aug_composed = al.Compose([aug_guass, aug_rgbshit])
        return aug_composed(image=image_arr)["image"]

    global _f_randomize_rgb_image
    _f_randomize_rgb_image = randomize_rgb_image


def configure_gray_image_randomizer():
    def randomize_gray_image(image_arr: np.ndarray):
        aug_guass = al.GaussNoise(p=1)
        aug_composed = al.Compose([aug_guass])
        return aug_composed(image=image_arr)["image"]

    global _f_randomize_gray_image
    _f_randomize_gray_image = randomize_gray_image


def configure_depth_image_randomizer(depth_shift_std=0.3, depth_noise_std=0.01):
    def randomize_depth_image(image_arr: np.ndarray):
        shift = np.random.randn() * depth_shift_std
        noise = np.random.randn(*image_arr.shape) * depth_noise_std
        image_arr_out = image_arr + noise + shift
        return image_arr_out

    global _f_randomize_depth_image
    _f_randomize_depth_image = randomize_depth_image


configure_rgb_image_randomizer()
configure_depth_image_randomizer()
configure_gray_image_randomizer()
