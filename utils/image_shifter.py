from typing import List

from PIL import Image
import numpy as np
import os
import time
import glob


def shift_image(image_name: str, shift_vector: tuple, save_file_name=None):
    if not os.path.exists('shifted_images'):
        os.mkdir('shifted_images')
    image = Image.open(image_name)
    shift_x, shift_y = shift_vector
    shifted_image = image.transform(image.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
    if save_file_name is None:
        new_file_name = f"{os.path.splitext(image_name)[0]}-shifted.png"
        shifted_image.save('shifted_images/' + new_file_name)
    else:
        shifted_image.save('shifted_images/' + save_file_name)


def batch_shift_images(image_names: List[str], shift_vector: tuple):
    print("Starting batch job")
    start_time = time.time()
    for idx, image_name in enumerate(image_names):
        shift_image(image_name, shift_vector)
    duration = time.time() - start_time
    print(f"Finished job of shifting {len(image_names)} in {duration} seconds")


class RuntimeShifter:
    shift_vector = (-15, 15)

    def __init__(self, shift_vector):
        self.shift_vector = shift_vector

    def shift_image(self, np_image):
        image = Image.fromarray(np_image)
        shift_x, shift_y = self.shift_vector
        shifted_image = image.transform(image.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
        return np.array(shifted_image)


if __name__ == "__main__":
    files_to_shift = glob.glob("*.png")
    batch_shift_images(files_to_shift, (-20, -10))

