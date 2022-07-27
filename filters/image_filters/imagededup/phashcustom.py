from pathlib import PurePath, Path
from typing import List, Union, Tuple

import numpy as np
from PIL import Image
import os

from filters.image_filters.imagededup.hashing import PHash
from typing import Callable, Dict, List, Tuple, Optional


IMG_FORMATS = ['JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF', 'WEBP']

def preprocess_image_fixed(
    image, target_size: Tuple[int, int] = None, grayscale: bool = False
) -> np.ndarray:
    """
    Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed
    image.
    Args:
        image: numpy array or a pillow image.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
    Returns:
        A numpy array of the processed image.
    """
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError('Input is expected to be a numpy array or a pillow object!')

    if target_size:
        image_pil = image_pil.resize(target_size, Image.Resampling.LANCZOS)

    if grayscale:
        image_pil = image_pil.convert('L')

    return np.array(image_pil).astype('uint8')

import filters.image_filters.imagededup.image_utils
filters.image_filters.imagededup.image_utils.preprocess_image = preprocess_image_fixed
from filters.image_filters.imagededup.image_utils import load_image


def _check_3_dim(image_arr_shape: Tuple) -> None:
    assert image_arr_shape[2] == 3, (
        f'Received image array with shape: {image_arr_shape}, expected image array shape is '
        f'(x, y, 3)'
    )


def _add_third_dim(image_arr_2dim: np.ndarray) -> np.ndarray:
    image_arr_3dim = np.tile(
        image_arr_2dim[..., np.newaxis], (1, 1, 3)
    )  # convert (x, y) to (x, y, 3) (grayscale to rgb)
    return image_arr_3dim


def _raise_wrong_dim_value_error(image_arr_shape: Tuple) -> None:
    raise ValueError(
        f'Received image array with shape: {image_arr_shape}, expected number of image array dimensions are 3 for '
        f'rgb image and 2 for grayscale image!'
    )


def check_image_array_hash(image_arr: np.ndarray) -> None:
    image_arr_shape = image_arr.shape
    if len(image_arr_shape) == 3:
        _check_3_dim(image_arr_shape)
    elif len(image_arr_shape) > 3 or len(image_arr_shape) < 2:
        _raise_wrong_dim_value_error(image_arr_shape)


class PHashCustom(PHash):
    """
    PHasher with extended IMG_FORMATS
    """
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
    
    def encode_image(
        self, image_file=None, image_array: Optional[np.ndarray] = None, grayscale: bool = False
    ) -> str:
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=grayscale,
                    img_formats=IMG_FORMATS
                )

            elif isinstance(image_array, np.ndarray):
                check_image_array_hash(image_array)  # Do sanity checks on array
                image_pp = preprocess_image_fixed(
                    image=image_array, target_size=self.target_size, grayscale=grayscale
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None