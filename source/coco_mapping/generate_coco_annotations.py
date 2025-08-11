import json
import os

import cv2
import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from coco_mapping.utils_coco import new_coco_dataset


def generate_coco_annotations(dataset, annotation_path, coco_mapping_list, append_to_file:False):

    # Process train then test split of images
    for split, split_list in dataset.split_dict.items():
    #for split in ['train', 'test']:
        dataset_file = os.path.join(annotation_path,
                                    f"instances_{split}.json")
        if append_to_file:
            with open(dataset_file, 'r') as f:
                coco_dataset = json.load(f)
            image_list = dataset.get('images', [])
            ann_list = dataset.get('annotations', [])
            image_id = image_list[-1]['id'] + 1
            ann_id = ann_list[-1]['id'] + 1
        else:
            coco_dataset = new_coco_dataset(coco_mapping_list)
            image_list, ann_list = list(), list()
            image_id, ann_id = 1, 1

    # Process image list
        for i in tqdm(range(len(split_list))):
            # Get image data
            img_data, image_annotations, img_path = dataset.get_image_data(split, i)
            img_data['id'] = img_data['id'] if isinstance(img_data['id'], int) else image_id
            image_list.append(img_data)

            # Create coco annotations (and text file)
            text_data = []
            for a in image_annotations:
                polygon = [] if 'polygon' not in a else a['polygon']
                annotation = {'category_id': a['category'],
                              'image_id': img_data['id'],
                              'id': ann_id,
                              'iscrowd': 0,
                              'ignore': 0,
                              'segmentation': [],
                              'semi': a.get('semi', False)}
                if 'bbox' in a:
                    file_line = f"{a['category']}, {int(a['bbox'][0])}, {int(a['bbox'][1])}, {int(a['bbox'][2])}, {int(a['bbox'][3])}\n"
                    text_data.append(file_line)
                    annotation['bbox'] = [int(b) for b in a['bbox']]
                    annotation['area'] = int(a['bbox'][2] * a['bbox'][3])
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
                    semi = a.get('semi', False)
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
                        image = cv2.drawContours(image, [np.array(a['polygon'])], 0, (255, 255, 255), 1)
                cv2.imwrite(output_name, image)

            image_id += 1

        # Add to dataset
        coco_dataset['images'] = image_list
        coco_dataset['annotations'] = ann_list

        # Save dataset
        with open(dataset_file, "w") as fp:
            json.dump(coco_dataset, fp)
