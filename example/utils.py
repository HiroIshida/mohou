import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def add_text_to_image(image: np.ndarray, text: str, color: str):

    def canvas_to_ndarray(fig, resize_pixel=None):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if resize_pixel is None:
            return data
        img = Image.fromarray(data)
        img_resized = img.resize(resize_pixel)
        data_resized = np.asarray(img_resized)
        return data_resized

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.text(7, 1, text, fontsize=25, color=color, verticalalignment='top')
    fig.canvas.draw()
    fig.canvas.flush_events()
    return canvas_to_ndarray(fig)
