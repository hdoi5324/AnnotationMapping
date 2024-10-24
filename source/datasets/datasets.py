import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F

class SourceData:
    def __init__(self):
        self.test_split = 0.3
        self.split_dict = {}
        self.image_id = 0
        self.ann_id = 0

    def _get_image(self, split, no):
        return self.split_dict[split][no]


class BlenderData(SourceData):
    def __init__(self, image_dir, image_template, opt):
        super(BlenderData, self).__init__()
        self.opt = opt
        self.test_split = self.opt.test_split
        self.imgs = glob(image_template)
        if self.test_split <= 0.0:
            self.split_dict['train'], self.split_dict['test'] = self.imgs, []
        else:
            self.split_dict['train'], self.split_dict['test'] = train_test_split(self.imgs, test_size=self.test_split)
        self.categories_of_interest = self.opt.blender_ooi

    def get_image_data(self, split, no, image_dir):
        img_path = self._get_image(split, no)

        img_dir, img_filename = os.path.split(img_path)
        img = read_image(img_path)
        image_data = {'license': 1,
                      'file_name': img_filename,
                      'height': img.shape[1],
                      'width': img.shape[2],
                      'id': self.image_id}

        ann_dir = os.path.join(img_dir, "../ann")
        mask_path = os.path.join(ann_dir, f"instance_segmaps_{img_filename[6:-4]}.npy")
        mask = np.load(mask_path)
        obj_ids = np.unique(mask)

        # Remove background - category 0
        masks = torch.Tensor(np.stack([mask == id for id in obj_ids], axis=0)) > 0

        # Load instance to category mappings
        read_dict = np.load(os.path.join(ann_dir, f"idx_class_map_{img_filename[6:-4]}.npy"),
                            allow_pickle='TRUE').item()
        all_boxes = masks_to_boxes(masks)

        # Get a list of category and bounding boxes.
        image_annotations = []
        for j, obj_id in enumerate(obj_ids):
            category = read_dict[obj_id]
            if category in self.categories_of_interest:
                mask = masks[j]
                #drawn_mask = draw_segmentation_masks(img, mask, colors="blue")
                #show(drawn_mask)
                #plt.show()
                coord = [int(c) for c in all_boxes[j]]
                # Update to from max x, max y to width, height
                coord[2] = coord[2] - coord[0]
                coord[3] = coord[3] - coord[1]

                # Add 5% all around
                buffer = 0.05
                x_buffer = int((coord[2] - coord[0]) * buffer)
                y_buffer = int((coord[3] - coord[1]) * buffer)
                coord[0] = max(0, coord[0] - x_buffer)
                coord[1] = max(0, coord[1] - y_buffer)
                coord[2] = min(coord[2]+ 2*x_buffer, img.shape[2])
                coord[3] = min(coord[3] + 2*y_buffer, img.shape[1])

                ann_data = {'category': category,
                            'semi': False,
                            'bbox': coord,
                            'source': "blender"
                            }
                image_annotations.append(ann_data)
                self.ann_id += 1
        self.image_id += 1

        return image_data, image_annotations, img_path

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
