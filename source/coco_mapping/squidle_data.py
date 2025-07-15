import os
from collections import OrderedDict
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from .datasets import SourceData
from sqapi.media import SQMediaObject

from .squidle_connection import SquidleConnection
from .utils_squidle import get_image_name

class SquidleData(SourceData):
    def __init__(self, opt, image_dir=None, subdir_paths=None):
        super(SquidleData, self).__init__()
        self.opt = opt
        if self.opt.test_split is not None:
            self.test_split = self.opt.test_split
        self.sq_id_to_cat_id = {cat: i + 1 for i, cat in enumerate(opt.squidle_mapping)}
        self.squidle_connection = SquidleConnection(api_key=self.opt.api_token, host=self.opt.url)
        self.subdir_paths = subdir_paths

        if image_dir is not None:  # if there's an image_dir get annotations/media for creating coco annotations
            self.image_dir = image_dir
            os.makedirs(image_dir, exist_ok=True)

            self.media_dict = OrderedDict()
            self.annotations_by_media_id = defaultdict(list)

            # 1) Get annotations and media ready to create coco annotations
            # Get relevant annotations
            exclude_annotation_set_ids = [] if self.opt.exclude_annotation_set_ids is None else list(self.opt.exclude_annotation_set_ids)
            needs_review_annotation_set_ids = [] if self.opt.needs_review_annotation_set_ids is None else list(self.opt.needs_review_annotation_set_ids)

            ann_count, anns, retrieved_ann_sets = self.get_annotations_by_media_id(list(self.opt.annotation_set_ids),
                                                                                   label_ids=list(
                                                                                       self.sq_id_to_cat_id.keys()),
                                                                                   needs_review_annotation_set_ids=
                                                                                       needs_review_annotation_set_ids,
                                                                                   inclusive=self.opt.inclusive,
                                                                                   exclude_annotation_sets=
                                                                                       exclude_annotation_set_ids)
            print(
                f"Found {ann_count} annotations for labels {list(self.sq_id_to_cat_id.keys())}  {'including' if self.opt.inclusive else 'excluding'} annotation sets {list(self.opt.annotation_set_ids)}")

            # Remove media_ids for any annotations that should be excluded.
            if not self.opt.inclusive:
                exclude_annotation_set_ids += list(self.opt.annotation_set_ids)
                # todo: Not sure of the logic here.  Why remove media ids from excluded annotation sets?
            media_ids_to_ignore = self.squidle_connection.get_media_ids_for_annotation_set_ids(
                exclude_annotation_set_ids)
            media_ids_to_remove = set(self.annotations_by_media_id.keys()).intersection(
                set(media_ids_to_ignore))
            for id in media_ids_to_remove:
                del self.annotations_by_media_id[id]
            print(f"Removing media from these annotation sets {exclude_annotation_set_ids} - media ids {media_ids_to_remove} leaving {len(self.annotations_by_media_id)} images with annotations.")

            # Get media items and save images with annotations and a ratio of images with no annotations
            if self.opt.ratio_non_annotated_images > 0:
                for id in retrieved_ann_sets:
                    # Get the media for the media_collection associated with the annotation sets retrieved
                    ann_set = self.squidle_connection.get_annotation_sets(annotation_set_id=id)
                    media_collection_id = ann_set.get("objects")[0].get("media_collection").get("id")
                    media_count = self.save_media_collection_data(media_collection_id,
                                                                  no_annotation_ratio=self.opt.ratio_non_annotated_images,
                                                                  media_ids_to_ignore=media_ids_to_ignore,
                                                                  results_per_page=1000)
                    print(
                        f"PROCESSED: {media_count} media items from annotation set {id} and media collection {media_collection_id}.")
            else:
                print(f"Only saving images for selected annotations.")
                media_list = self.squidle_connection.get_media_obj_for_media_ids(
                    [k for k in self.annotations_by_media_id.keys()])
                self.save_media_list_data(media_list)

            # 2. Split into test/train as required
            self.imgs = [k for k in self.media_dict.keys()]
            if self.test_split >= 1.0:
                self.split_dict['train'], self.split_dict['test'] = [], self.imgs
            elif self.test_split <= 0.0:
                self.split_dict['train'], self.split_dict['test'] = self.imgs, []
            else:
                self.split_dict['train'], self.split_dict['test'] = [], []
                train_ann_count = int(len(self.annotations_by_media_id.keys()) * (1 - self.test_split))
                ann_count = 0
                for path, data in self.media_dict.items():
                    if len(data[1]) > 0:
                        ann_count += 1
                    if ann_count < train_ann_count:
                        self.split_dict['train'].append(path)
                    else:
                        self.split_dict['test'].append(path)
                # Use below to get a random split rather than in order of path
                #self.split_dict['train'], self.split_dict['test'] = train_test_split(list(self.media_dict.keys()), test_size=self.test_split)

    def save_media_collection_data(self, media_collection_id, media_ids_to_ignore=[], results_per_page=200,
                                   no_annotation_ratio=500):
        """ Saves images from the media_collection for images with annotations and a selection of images with no annotations.
        Saves media to output_dir and also collects annotations for the media item.  Some media items may
        not have any annotations.
        Ignores media_ids in media_ids_to_ignore.  This is to ensure datasets and separate between train, target, test and val.
        Uses no_annotation_ratio to select some images so the dataset is not too large."""

        # Get all media_obj items for given media_collection_id
        media_count = 0
        media_list = self.squidle_connection.get_media_obj_for_media_collection_id(media_collection_id,
                                                                                   results_per_page=results_per_page)
        for m_obj in media_list:  # Iterate through each image
            media_id = m_obj.get("id")
            if media_id not in media_ids_to_ignore:
                media_count += 1
                # Check if in annotation dict
                annotations = self.annotations_by_media_id.get(media_id, None)
                if annotations is not None:
                    image_filepath = self.save_media_image(m_obj)
                    self.media_dict[image_filepath][1].extend(
                        annotations)  # add annotations to what's there already for this media_id
                if no_annotation_ratio > 0 and media_count % no_annotation_ratio == 0:
                    _ = self.save_media_image(m_obj)
            else:
                print(f"Ignoring {media_id}")

        return media_count

    def save_media_list_data(self, media_list, semi=False):
        " Not used at the moment."
        media_count = 0
        for m_obj in media_list:  # Iterate through each image
            media_count += 1
            # Check if in annotation dict
            media_id = m_obj.get("id")
            annotations = self.annotations_by_media_id.get(media_id, None)
            if annotations is not None:
                image_filepath = self.save_media_image(m_obj)
                self.media_dict[image_filepath][1].extend(
                    annotations)  # add annotations to what's there already for this media_id
            # todo: should we save media that doessn't have an annotation here?  This is the out of annotation set images.
            _ = self.save_media_image(m_obj)
        return media_count

    def save_media_image(self, media_item):
        # Get image and save
        media_url = media_item.get('path_best')
        image_name = get_image_name(media_url)
        image_filepath = os.path.join(self.image_dir, image_name)
        if image_filepath not in self.media_dict.keys():
            media_type = media_item.get("media_type", {}).get("name")
            mediaobj = SQMediaObject(media_url, media_type=media_type, media_id=media_item.get('id'))
            self.media_dict[image_filepath] = [media_item, []]
            _ = self.get_image(mediaobj, image_filepath)
        return image_filepath

    def get_image(self, mediaobj, image_filepath):
        if not os.path.exists(image_filepath):
            image_data = mediaobj.data()
            img = Image.fromarray(image_data)  # has already been padded, so will return padded image
            cv2.imwrite(image_filepath, np.array(img))
            print(f"Saving {image_filepath}")
        return image_filepath


    def get_bbox_in_pixels(self, x, y, polygon, width, height, buffer=0.05):
        # Return the bounding box based on the max and min x and y coordinates
        return get_bbox_in_pixels(x, y, polygon, width, height, buffer=buffer)

    def get_point_in_pixels(self, x, y, width, height):
        x = int(x * width)
        y = int(y * height)
        return [x, y]

    def get_bbox_from_point_in_pixels(self, x, y, width, height, buffer=0.035):
        # Creates an estimated bounding box around the point.
        return get_bbox_from_point_in_pixels(x, y, width, height, buffer=buffer)

    def get_image_data(self, split, i):
        # Copy image to split directory (test or train)
        img_path = self._get_image(split, i)
        self.copy_image_to_split(split, img_path, os.path.split(img_path)[1])

        [media, anns] = self.media_dict[img_path]
        img = Image.open(img_path)
        # if len(anns) == 0:
        #    print("No annotations ")
        img_data = {'file_name': os.path.basename(img_path),
                    'height': img.size[1],
                    'width': img.size[0],
                    'id': media['id'],
                    }

        image_annotations = []
        # Decide if bbox or point annotations.  Must be the same for the whole image.
        if len(anns) > 0:
            for a in anns:
                category = self.sq_id_to_cat_id.get(a['label']['id'], 0)
                point = a['point']
                polygon_ann = len(point['data'].get('polygon', [])) >= 3
                semi = a.get('semi', False)
                ann_data = {'category': category,
                            'semi': semi,
                            'source': 'squidle'
                            }
                # Bbox annotation
                bbox = None
                if polygon_ann:
                    bbox = self.get_bbox_in_pixels(point['x'],
                                                   point['y'],
                                                   point['data']['polygon'],
                                                   img.size[0],
                                                   img.size[1])
                    polygon_in_px = [[int((p[0] + point['x']) * img.size[0]), int((p[1] + point['y']) * img.size[1])]
                                     for p in point['data']['polygon']]
                    ann_data['polygon'] = polygon_in_px
                elif not polygon_ann:
                    # Point annotation estimation around point.  Rough
                    # todo: Change this to some other estimation method
                    bbox = self.get_bbox_from_point_in_pixels(point['x'],
                                                              point['y'],
                                                              img.size[0],
                                                              img.size[1])
                ann_data['bbox'] = bbox
                if bbox is not None:
                    image_annotations.append(ann_data)

        return img_data, image_annotations, img_path

    def get_annotations_by_media_id(self, annotation_set_ids, label_ids,
                                    needs_review_annotation_set_ids=[],
                                    exclude_annotation_sets=[],
                                    inclusive=True, results_per_page=200):
        """Get annotations and media details based on queries using label_ids, annotation set ids, inclusive flag (include or exclude annotation set ids),
        list of annotation sets that need to be filtered by review_flag = True, and annotation sets to exclude completely.

        First get all the relevant annotations.

        Then put in dictionary by media_id checking if the point has been added already based on annotation_id."""

        retrieved_annotation_set_ids = set()
        # Get annotations for the labels and annotation set ids. Inclusive or exclusive based on inclusive flag.
        # non_review_annotation_sets = list(set(annotation_set_ids) - set(needs_review_annotation_set_ids))
        non_review_annotation_sets = list(set(annotation_set_ids)) #- set(needs_review_annotation_set_ids))
        annotation_list = self.squidle_connection.get_all_annotations_from_set(non_review_annotation_sets,
                                                                               label_ids=label_ids,
                                                                               include_annotation_sets=inclusive,
                                                                               exclude_annotation_sets=exclude_annotation_sets, #+ needs_review_annotation_set_ids,
                                                                               needs_review=False,
                                                                               results_per_page=results_per_page)

        # Handle review_flag annotation sets where we only want the annotations with review_flag set to True
        if inclusive:
            review_annotation_sets = list(set(annotation_set_ids).intersection(set(needs_review_annotation_set_ids)))
        else:
            review_annotation_sets = list(set(needs_review_annotation_set_ids) - set(annotation_set_ids))
        if len(review_annotation_sets) > 0:
            annotation_list += self.squidle_connection.get_all_annotations_from_set(review_annotation_sets,
                                                                                    label_ids=label_ids,
                                                                                    include_annotation_sets=True,
                                                                                    exclude_annotation_sets=exclude_annotation_sets,
                                                                                    needs_review=True,
                                                                                    results_per_page=results_per_page)

        ann_count = 0
        for ann in annotation_list:
            # Keep annotations with an x,y coordinate.  They'll either be a point or have a bbox
            media_id = ann["point"]['media_id']
            if ann["point"]["x"] is not None:
                existing_ann_ids = [ann['id'] for ann in self.annotations_by_media_id[media_id]]
                if ann['id'] not in existing_ann_ids:
                    self.annotations_by_media_id[media_id].append(ann)
                    ann_count += 1
                    retrieved_annotation_set_ids.add(ann['annotation_set_id'])
            else:
                print(ann["point"])
        return ann_count, annotation_list, list(retrieved_annotation_set_ids)


def get_bbox_in_pixels(x, y, polygon, width, height, buffer=0.05):
    # Return the bounding box based on the max and min x and y coordinates

    min_x = (np.min([p[0] for p in polygon]) + x) * width
    max_x = (np.max([p[0] for p in polygon]) + x) * width
    min_y = (np.min([p[1] for p in polygon]) + y) * height
    max_y = (np.max([p[1] for p in polygon]) + y) * height
    width_buffer = (max_x - min_x) * buffer
    height_buffer = (max_y - min_y) * buffer
    min_x = int(min_x - width_buffer)
    max_x = int(max_x + width_buffer)
    min_y = int(min_y - height_buffer)
    max_y = int(max_y + height_buffer)
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def get_bbox_from_point_in_pixels(x, y, width, height, buffer=0.035):
    # Creates an estimated bounding box around the point.
    min_x = int((x - buffer) * width)
    max_x = int((x + buffer) * width)
    min_y = int((y - buffer) * height)
    max_y = int((y + buffer) * height)
    return [min_x, min_y, max_x - min_x, max_y - min_y]