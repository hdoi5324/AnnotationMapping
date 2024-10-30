import json
import os

import cv2
import hydra
import matplotlib as mpl
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from datasets.utils_coco import new_coco_dataset
from utils.set_random_seed import set_random_seed

"""
Generate coco annotation files and test/train split for either squidle images
or blender generated images."""


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(opt: DictConfig) -> None:
    set_random_seed(opt.seed)
    output_dir = os.path.join(os.path.expanduser(opt.output_dir), opt.dataset.folder)
    os.makedirs(output_dir, exist_ok=True)
    subdir_paths = {}
    for subdir in ["annotations", f"test{opt.dataset.year}", f"train{opt.dataset.year}"]:
        p = os.path.join(output_dir, subdir)
        os.makedirs(p, exist_ok=True)
        subdir_paths[subdir] = p
    if opt.dataset.datatype == "Infinigen":
        from datasets.infinigen import InfinigenData
        dataset = InfinigenData(output_dir, opt=opt.dataset, subdir_paths=subdir_paths)
        # Create dir for no water images
        for subdir in [f"test{opt.dataset.year}_nowater", f"train{opt.dataset.year}_nowater"]:
            p = os.path.join(output_dir, subdir)
            os.makedirs(p, exist_ok=True)   
    elif opt.dataset.datatype == "Squidle":
        from datasets.squidle_data import SquidleData
        dataset = SquidleData(opt.dataset, image_dir=output_dir, subdir_paths=subdir_paths)
    else:
        print(f"Not implemented: {opt.dataset.datatype}")
    generate_coco_annotations(dataset, subdir_paths, output_dir, opt)


def generate_coco_annotations(dataset, subdir_paths, output_dir, opt):
    image_id, ann_id = 0, 0

    # Process train then test split of images
    for split in ['train', 'test']:
        split_dataset = new_coco_dataset(opt.dataset.coco_mapping)
        image_list, ann_list = list(), list()

        # Process image list
        for i in tqdm(range(len(dataset.split_dict[split]))):

            # Get image data
            img_data, image_annotations, img_path = dataset.get_image_data(split, i)
            image_list.append(img_data)

            # Create coco annotations (and text file)
            text_data = []
            obj_id = 1
            for a in image_annotations:
                polygon = [] if 'polygon' not in a else a['polygon']
                annotation = {'category_id': a['category'],
                              'image_id': img_data['id'],
                              'id': ann_id,
                              'iscrowd': 0,
                              'ignore': 0,
                              'segmentation': [],
                              'semi': a['semi']}
                if 'bbox' in a:
                    file_line = f"{a['category']}, {int(a['bbox'][0])}, {int(a['bbox'][1])}, {int(a['bbox'][2])}, {int(a['bbox'][3])}\n"
                    text_data.append(file_line)
                    annotation['bbox'] = list(a['bbox'])
                    annotation['area'] = a['bbox'][2] * a['bbox'][3]
                if 'point' in a:
                    annotation['point'] = list(a['point'])
                if len(polygon) >= 3:
                    annotation['polygon'] = list(polygon)

                ann_list.append(annotation)
                ann_id += 1

            if len(image_annotations) > 0:
                # Save text file with bounding boxes
                box_path = f"{img_path[:-4]}.txt"
                with open(box_path, 'w+') as f:
                    for line in text_data:
                        f.write(line)

                # Save image with bounding boxes and point annotations
                output_name = f"{img_path[:-4]}_viz.jpg"
                image = cv2.imread(img_path)
                colours = [(int(255 * c[0]), int(255 * c[1]), int(255 * c[2])) for c in mpl.colormaps['tab10'].colors]
                for a in image_annotations:
                    cat = a['category']
                    semi = a['semi']
                    colour = colours[cat % len(colours)] if not semi else colours[-1]
                    if 'bbox' in a:
                        coords = a['bbox']
                        [x, y, width, height] = coords
                        x_min = int(x)
                        y_min = int(y)
                        x_max = x_min + int(width)
                        y_max = y_min + int(height)
                        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour, 4)
                    elif 'point' in a:
                        coords = a['point']
                        x = int(coords[0])
                        y = int(coords[1])
                        image = cv2.circle(image, (x, y), 8, colour, 16)
                    if 'polygon' in a:
                        image = cv2.drawContours(image, [np.array(a['polygon'])], 0, (255, 255, 255), 2)
                cv2.imwrite(output_name, image)

            image_id += 1

        # Add to dataset
        split_dataset.dataset['images'] = image_list
        split_dataset.dataset['annotations'] = ann_list

        # Save dataset
        dataset_file = os.path.join(subdir_paths["annotations"],
                                    f"instances_{split}{split_dataset.dataset['info']['year']}.json")
        with open(dataset_file, "w") as fp:
            json.dump(split_dataset.dataset, fp)


if __name__ == "__main__":
    main()
