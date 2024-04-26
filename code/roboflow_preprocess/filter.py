import json

import yaml
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.path_util import rebase


@dataclass
class BoundingBox:
    first_row: int
    last_row: int
    first_col: int
    last_col: int

    image_shape: tuple[int, int]

    @property
    def area(self) -> int:
        return (self.last_row - self.first_row) * (self.last_col - self.first_col)

    def to_json(self):
        return {
            'first_row': self.first_row,
            'last_row': self.last_row,
            'first_col': self.first_col,
            'last_col': self.last_col,
            # 'image_shape': self.image_shape,
        }


bboxes = {}


def get_bounding_box(path: Path) -> BoundingBox:
    """
    This function receives a picture path,
    loads the image,
    and returns the bounding box of the non-black pixels.
    """

    if path in bboxes:
        return bboxes[path]

    # Load the image (PIL + numpy)
    img = np.array(Image.open(path))

    # Find first row with non-black pixels
    first_row = 0
    while np.all(img[first_row, :, 0] == 0):
        first_row += 1

    # Find last row with non-black pixels
    last_row = img.shape[0] - 1
    while np.all(img[last_row, :, 0] == 0):
        last_row -= 1

    # Find first column with non-black pixels
    first_col = 0
    while np.all(img[:, first_col, 0] == 0):
        first_col += 1

    # Find last column with non-black pixels
    last_col = img.shape[1] - 1
    while np.all(img[:, last_col, 0] == 0):
        last_col -= 1

    # Return the bounding box
    result = BoundingBox(first_row, last_row, first_col, last_col, img.shape[:1])
    bboxes[path] = result
    return result


def get_original(path1: Path, path2: Path) -> Path:
    """
    This function receives two picture paths,
    loads the images,
    and decides which one is rotated,
    and returns the other (original) image.

    It works if the background on both images is black.
    One of the pictures has more black pixels in its first and end rows than the other.
    """

    # Get the bounding boxes
    bb1 = get_bounding_box(path1)
    bb2 = get_bounding_box(path2)

    assert bb1.image_shape == bb2.image_shape

    # Calculate effective area of the image
    area1 = bb1.area
    area2 = bb2.area

    # Return the original image (which has smaller area)
    if area1 < area2:
        return path1
    else:
        return path2


def find_pairs(list_of_paths: list[Path]) -> list[list[Path]]:
    """
    This function receives a list of image paths,
    and finds similarly named files.

    The file names are of pattern sometext.sometext.somehash.extension
    Two file names are considered similar, if they have the same sometext.sometext.
    """

    # Create a dictionary to store the pairs
    pairs = {}

    # Iterate over the paths
    for path in list_of_paths:

        # Split the file name
        name = path.stem.split(".")
        restored_name = '.'.join(name[:-2])
        assert restored_name != ''

        # If the first two parts of the name are not in the dictionary, add them
        if restored_name not in pairs:
            pairs[restored_name] = []
        pairs[restored_name].append(path)

    # Return the pairs
    return list(pairs.values())


def dedupliate(pair: list[Path]):
    if len(pair) == 2:
        return get_original(*pair)
    else:
        return get_original(pair[0], dedupliate(pair[1:]))


def main():
    pairs = find_pairs(list(Path('/data/Roboflow/New Final Dataset').glob('**/train/*.jpg')))
    result = []
    for i, pair in enumerate(tqdm(pairs)):
        original = dedupliate(pair)
        result.append({
            'path': str(original),
            'bbox': get_bounding_box(original).to_json()
        })

        # new_path = rebase(Path('/data'), f'/out/preprocess/filter', original)
        # new_path.parent.mkdir(parents=True, exist_ok=True)
        # new_path.write_bytes(original.read_bytes())

    with open('/out/preprocess/filter/result.yaml', 'w') as f:
        yaml.dump(result, f)


if __name__ == '__main__':
    main()
