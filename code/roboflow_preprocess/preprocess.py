from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

from common.path_util import data_dir, preprocess_out_dir
from common.plot_util import save_next


def main():
    first = True

    for dataset in [
        'vzrad2',
        'New Final Dataset'
    ]:
        for version in Path(data_dir / 'Roboflow' / dataset).iterdir():
            for split in version.iterdir():
                if split.is_dir():
                    (preprocess_out_dir / 'Roboflow' / dataset / version.name / split.name).mkdir(parents=True,
                                                                                                  exist_ok=True)
                    for file in split.iterdir():
                        if file.suffix == '.json':
                            coco = COCO(file)
                            cat_ids = coco.getCatIds(catNms=['Caries' if dataset == 'vzrad2' else 'caries'])
                            ids = coco.getImgIds(catIds=cat_ids)
                            for id_ in ids:
                                anns_ids = coco.getAnnIds(imgIds=id_, catIds=cat_ids, iscrowd=None)
                                anns = coco.loadAnns(anns_ids)

                                path = coco.loadImgs(id_)[0]['file_name']

                                try:
                                    mask = coco.annToMask(anns[0])
                                    for i in range(1, len(anns)):
                                        mask |= coco.annToMask(anns[i])
                                except IndexError:
                                    print(f'Error with {path}')
                                    continue
                                mask_image = Image.fromarray((np.clip(mask * 255, 0, 255)).astype(np.uint8))

                                mask_image.save(
                                    preprocess_out_dir / 'Roboflow' / dataset / version.name / split.name / f'{Path(path).stem}.png'
                                )

                                if first:
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                                    img1 = np.array(Image.open(split / path))
                                    print(img1.shape)
                                    print(split / path)
                                    ax1.imshow(img1)
                                    ax2.set_title(f'Image')

                                    img2 = mask
                                    ax2.imshow(img2)
                                    ax2.set_title(f'Mask')

                                    save_next(fig, 'test_preprocess', with_fig_num=False)
                                    first = False
                            break


if __name__ == '__main__':
    main()
