import albumentations as al
import numpy as np
from typing import Optional, Callable

# Becuase python does not have multiple-dispach, considering extensability
# we have to adopt this anti-pattern.
_f_randomize_rgb_image: Optional[Callable[[np.ndarray], np.ndarray]] = None


def configure_rgb_image_randomizer(rgb_shift_limit=40):
    def randomize_rgb_image(image_arr: np.ndarray):
        aug_guass = al.GaussNoise(p=1)
        sl = rgb_shift_limit
        aug_rgbshit = al.RGBShift(r_shift_limit=sl, g_shift_limit=sl, b_shift_limit=sl)
        aug_composed = al.Compose([aug_guass, aug_rgbshit])
        return aug_composed(image=image_arr)['image']
    global _f_randomize_rgb_image
    _f_randomize_rgb_image = randomize_rgb_image


configure_rgb_image_randomizer()
