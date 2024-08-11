"""
A handy module that will download Google Quickdraw dataset using the quickdraw Python module.
"""

from typing import Tuple

from quickdraw import QuickDrawDataGroup

from functions import make_data_dirs

QUICKDRAW_CACHE_DEFAULT: str = "./.quickdrawcache"
QUICKDRAW_IMAGE_DEFAULT: str = "data/quickdraw"


def setup_data_dirs() -> Tuple[str, str]:
    """Function to setup the data and cache directories
    Returns:
        Tuple[str, str]: Tuple containing the cache directory and image directory
    """
    cache_dir: str = input(
        f"Enter the cache directory. Leave blank to default to: {QUICKDRAW_CACHE_DEFAULT}")
    image_dir: str = input(
        f"Enter image directory. Leave blank to default to {QUICKDRAW_IMAGE_DEFAULT}")

    if cache_dir == "":
        cache_dir = QUICKDRAW_CACHE_DEFAULT

    if image_dir == "":
        image_dir = QUICKDRAW_IMAGE_DEFAULT

    return (cache_dir, image_dir)

def gen_class_images(class_names: list, download_cache_dir: str, image_dir: str,
                    stroke_widths: list, max_drawings: int = 1000,
                    drawing_size=(28, 28), recognized: bool = True,
                    img_ext: str = "png"):
    """Function to generate images for a class
    Args:
        class_name (str): The class name
        download_cache_dir (str): The download cache directory
        image_dir (str): The image directory
        max_drawings (int, optional): The maximum number of drawings. Defaults to 1000.
        drawing_size (tuple, optional): The drawing size. Defaults to (28, 28).
        recognized (bool, optional): If the drawing is recognized. Defaults to True.
        stroke_width (list, optional): The stroke width. Defaults to [3].
    """
    
    for class_name in class_names:
        qdg = QuickDrawDataGroup(class_name, max_drawings=max_drawings,
                                recognized=recognized, cache_dir=download_cache_dir)

        for i, drawing in enumerate(qdg.drawings):
            for width in stroke_widths:
                class_image_dir = f"{image_dir}/{class_name}"
                make_data_dirs(class_image_dir)
                filepath = f"{class_image_dir}/{class_name}_{i}_width_{width}.{img_ext}"
                drawing.get_image(stroke_width=width).resize(drawing_size).save(filepath)

def main():
    print("Welcome to the Google QuickDraw Data downloading utility!")
    cache_dir, image_dir = setup_data_dirs()

    # Make the data directories
    make_data_dirs(cache_dir, image_dir)

