import os
import cv2
from glob import glob
import json
import numpy as np
import torch
import shutil
from einops import pack, rearrange, repeat
import colorsys

from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from datasets.datasets import SourceData


class InfinigenData(SourceData):
    def __init__(self, output_dir, opt, subdir_paths, no_channel_splits=10):
        super(InfinigenData, self).__init__()
        self.opt = opt
        self.no_channel_splits = self.opt.get(no_channel_splits, no_channel_splits)
        self.output_dir = os.path.expanduser(output_dir)
        self.test_split = self.opt.test_split
        image_template = f"{self.output_dir}/*/frames/Image/*/*png"
        #todo: filter out images that don't have the supporting files
        self.imgs = glob(image_template)
        self.imgs = [p for p in self.imgs if os.path.exists(os.path.expanduser(p.replace("Image", "InstanceSegmentation").replace(".png", ".npy")))]
        self.imgs = [p for p in self.imgs if os.path.exists(os.path.expanduser(p.replace("Image", "ObjectSegmentation").replace(".png", ".npy")))]
        if self.test_split <= 0.0:
            self.split_dict['train'], self.split_dict['test'] = self.imgs, []
        else:
            self.split_dict['train'], self.split_dict['test'] = train_test_split(self.imgs, test_size=self.test_split)
        self.categories_of_interest = self.opt.ooi
        self.subdir_paths = subdir_paths


    def get_new_filename(self, img_path):
        img_dir, img_filename = os.path.split(img_path)
        img_dir_parts = img_dir.split('/')
        new_img_filename = f"{img_dir_parts[-5]}_{img_dir_parts[-4]}_{img_dir_parts[-1]}_{img_filename[:-4]}{img_filename[-4:]}"  # _{no:05d}
        return new_img_filename

    def get_image_data(self, split, no):
        # Generate annotation data using infinigen object and instance marks.
        # Based on infinigen and blenderproc approaches to segmentation

        # Copy image to split directory (test or train)
        img_path = self._get_image(split, no)
        new_img_filename = self.get_new_filename(img_path)
        subdir = self.copy_image_to_split(split, img_path, new_img_filename)

        # Copy No Water image too if it exists
        no_water_img_path = img_path.replace('Image', 'ImageNoWater')
        no_water_subdir = f"{subdir}_nowater"
        if os.path.exists(no_water_img_path) and not os.path.exists(os.path.join(no_water_subdir, os.path.split(img_path)[1])):
            shutil.copy(no_water_img_path, os.path.join(no_water_subdir, new_img_filename))


        # Get image data
        img = read_image(img_path)
        _, H, W = img.shape
        image_data = {'file_name': new_img_filename,
                      'height': img.shape[1],
                      'width': img.shape[2],
                      'id': self.image_id}

        # Using object and instance masks to find box annotations
        object_segmentation_mask = np.load(get_output_filename(img_path, "ObjectSegmentation", "npy"))
        instance_segmentation_mask = cv2.imread(get_output_filename(img_path, "InstanceSegmentation", "png"))
        #instance_segmentation_mask = np.round(instance_segmentation_mask / self.no_channel_splits, decimals=0).astype(np.uint8)

        #denoised_instance_seg = remove_noisy_pixels(instance_segmentation_mask, H, W)

        # Hierarchical clustering


        # Reduce colours using kmeans. Helps get rid of colours that are close to each other.
        # Note limited to 128 objects (clusters)
        #instance_segmentation_mask = kmeans_color_quantization(instance_segmentation_mask, clusters=128)

        # Collapse 3 channels to single unique number by multiplying together
        #instance_segmentation_mask = instance_segmentation_mask[:, :, 0] * instance_segmentation_mask[:, :, 1] * instance_segmentation_mask[:, :, 2]

        # Load instance to category mappings from infinigen outputs
        object_filename = get_object_filename(img_path)
        with open(object_filename, "r") as read_file:
            object_json = json.load(read_file)

        # Identify objects visible in the image
        obj_ids = np.unique(object_segmentation_mask)
        present_objects = [obj for obj in object_json.items() if (obj[1]['object_index'] in obj_ids)]

        # Iterate through each category OOI - AIM Get a dictionary of instance_id to category
        image_annotations = []
        for c, cat in enumerate(self.categories_of_interest):
            # Mask the pixels with any relevant object
            objects_to_highlight = [obj for obj in present_objects if (cat.lower() in obj[0].lower())]
            if len(objects_to_highlight) > 0:
                highlighted_pixels = should_highlight_pixel_fast(object_segmentation_mask,
                                                            np.array([o[1]['object_index'] for o in objects_to_highlight]))
                assert highlighted_pixels.dtype == bool

                # mask pixels not in object segmentation mask and find boxes
                # Assign unique colors to each object instance
                highlighted_instances = np.where(np.stack([highlighted_pixels] * 3, axis=-1), instance_segmentation_mask, 0)
                clustered_instance_seg = hierarchical_clustering(highlighted_instances, max_dist=1.8)
                bbox = compute_boxes_v2(clustered_instance_seg, highlighted_pixels)

                # Remove small boxes
                if len(bbox) > 0:
                    m = (bbox[:, 2]-bbox[:, 0]) * (bbox[:, 3]-bbox[:, 1]) > H*W*0.001
                    bbox = bbox[m]

                for coord in bbox:
                    coord = [int(cc) for cc in coord]
                    # Update to from max x, max y to width, height
                    coord[2] = coord[2] - coord[0]
                    coord[3] = coord[3] - coord[1]

                    # Add small percentage around bbox
                    buffer = 0.02
                    x_buffer = int(coord[2] * buffer)
                    y_buffer = int(coord[3] * buffer)
                    coord[0] = max(0, coord[0] - x_buffer)
                    coord[1] = max(0, coord[1] - y_buffer)
                    coord[2] = min(coord[2] + 2 * x_buffer, img.shape[2])
                    coord[3] = min(coord[3] + 2 * y_buffer, img.shape[1])

                    ann_data = {'category': c+1,
                                'semi': False,
                                'bbox': coord,
                                'source': "infinigen"
                                }
                    image_annotations.append(ann_data)
                    self.ann_id += 1
        self.image_id += 1

        return image_data, image_annotations, img_path

def remove_noisy_pixels(img_in, H, W, percent=0.0001):
    # Remove further noise where there are some stray pixel values with very small counts, by assigning them to
    # their closest (numerically, since this deviation is a result of some numerical operation) neighbor.
    b, counts = np.unique(img_in.reshape((-1, 3)), return_counts=True, axis=0)
    # Assuming the stray pixels wouldn't have a count of more than 0.5% of image size
    noise_vals = []
    for i in range(len(b)):
       if counts[i] <= H * W * percent:
           noise_vals.append(b[i].reshape((-1)))
    print(f"Removing {noise_vals} with pixel count less than {H * W * percent}")
    mask = np.zeros((H, W), dtype=bool)
    for point in noise_vals:
        new_mask = np.where(img_in[:, :, 0] == point[0], np.ones((H, W)).astype(bool), False) \
               & np.where(img_in[:, :, 1] == point[1], np.ones((H, W)).astype(bool), False) \
               & np.where(img_in[:, :, 2] == point[2], np.ones((H, W)).astype(bool), False)
        mask += new_mask
    mask = np.stack([np.logical_not(mask)] * 3, axis=-1)
    img_out = np.where(mask, img_in, -1000)
    return img_out


def get_output_filename(input_path, output_type, output_suffix, input_type="Image", input_suffix="jpg"):
    input_dir, input_filename = os.path.split(input_path)
    output_dir = input_dir.replace(input_type, output_type)
    output_filename = input_filename.replace(input_type, output_type)
    output_filename = output_filename[:-len(input_suffix)]+output_suffix
    return os.path.join(output_dir, output_filename)


def get_object_filename(img_path):
    # Load instance to category mappings
    exists = False
    tries = 200
    for _ in range(tries):
        object_file = get_output_filename(img_path, "Objects", "json")
        exists = os.path.exists(object_file)
        # Look for lower number filename
        if not exists:
            img_dir, img_filename = os.path.split(img_path)
            img_parts = os.path.basename(img_filename).split('_')
            previous_frame_no = int(img_parts[-2]) -1
            img_parts[-2] = f"{previous_frame_no:04d}"
            img_path = os.path.join(img_dir, '_'.join(img_parts))
    assert exists, f"Object file not found: {object_file}"
    return get_output_filename(img_path, "Objects", "json")

def should_highlight_pixel(arr2d, set1d):
    """Compute boolean mask for items in arr2d that are also in set1d"""
    H, W = arr2d.shape
    output = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            for n in range(set1d.size):
                output[i,j] = output[i,j] or (arr2d[i,j] == set1d[n])
    return output


def should_highlight_pixel_fast(arr2d, set1d):
    """Compute boolean mask for items in arr2d that are also in set1d"""
    img_in = arr2d
    H, W = arr2d.shape
    mask = np.zeros((H, W), dtype=bool)
    for point in set1d:
        new_mask = np.where(img_in[:, :] == point, np.ones((H, W)).astype(bool), False)
        mask += new_mask
    return mask

def arr2color(e):
    s = np.random.RandomState(np.array(e, dtype=np.uint32))
    return (np.asarray(colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)) * 255).astype(np.uint8)

def compute_boxes(indices, binary_tag_mask):
    """Compute 2d bounding boxes for highlighted pixels"""
    H, W = binary_tag_mask.shape
    indices = np.where(binary_tag_mask, indices, 0)
    num_u = len(np.unique(indices)) -1 #indices.max() #+ 1
    x_min = np.full(num_u, W-1, dtype=np.int32)
    y_min = np.full(num_u, H-1, dtype=np.int32)
    x_max = np.full(num_u, -1, dtype=np.int32)
    y_max = np.full(num_u, -1, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            idx = indices[y, x]
            tag_is_present = binary_tag_mask[y, x]
            if tag_is_present and idx != 0:
                x_min[idx] = min(x_min[idx], x)
                x_max[idx] = max(x_max[idx], x)
                y_min[idx] = min(y_min[idx], y)
                y_max[idx] = max(y_max[idx], y)
    return np.stack((x_min, y_min, x_max, y_max), axis=-1)

def compute_boxes_v2(indices, binary_tag_mask):
    """Compute 2d bounding boxes for highlighted pixels"""
    indices = np.where(binary_tag_mask, indices, 0)
    uniq_indices, uniq_counts = np.unique(indices, return_counts=True)
    bboxes = []
    confidence_scores = []
    for i, uniq_idx in enumerate(uniq_indices):
        if uniq_idx != 0 and uniq_counts[i] > 500:
            a = np.where(indices == uniq_idx)
            bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
            area = (bbox[2] - bbox[0])*(bbox[3]-bbox[1])
            pixels = uniq_counts[i]
            if pixels/area > 0.0:
                confidence = 0.7 + 0.3*(pixels/area)
                confidence_scores.append(confidence)
                bboxes.append(bbox)

    # NMS - reduce the number of overlapping boxes
    if len(bboxes) > 0:
        indices = cv2.dnn.NMSBoxes(bboxes=bboxes, scores=confidence_scores, score_threshold=0.7, nms_threshold=0.7)
        bboxes = [bboxes[i] for i in indices.flatten()]

    return np.vstack(bboxes) if len(bboxes) > 0 else bboxes

def is_in(element, test_elements, assume_unique=False, invert=False):
    """ As np.isin is only available after v1.13 and blender is using 1.10.1 we have to implement it manually. """
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)

def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i, v in enumerate(vec):
        if item == v:
            return i
    return -1
def hierarchical_clustering(image, max_dist=1.0, option='fast'):
    if option == 'simple':
        unique_points = np.unique(image.reshape((-1, 3)), axis=0)
    else:
        hashed_image = np.dot(image.reshape((-1, 3)).astype(np.uint32), [1, 256, 65536])
        unique_points = np.unique(hashed_image, axis=0)
        ignore = np.dot(np.array([-1000]*3).astype(np.uint32), [1, 256, 65536])
        unique_points = [p for p in unique_points if np.any(p != ignore)]
        flat_image = image.reshape((-1, 3))
        unique_points = [flat_image[np.argwhere(hashed_image == p)[0], :] for p in unique_points]
        #unique_points = [flat_image[find_first(p, hashed_image), :] for p in unique_points]
        unique_points = np.array(unique_points).squeeze()
            
    clustered_image = np.zeros(image.shape[:-1])
    H, W, _ = image.shape
    if len(unique_points) > 1:
        Z = linkage(unique_points,
                    method='complete',  # dissimilarity metric: max distance across all pairs of
                    # records between two clusters
                    metric='euclidean'
                    )  # you can peek into the Z matrix to see how clusters are merged at each iteration of the algorithm
        clusters = fcluster(Z, max_dist, criterion='distance')
    else:
        clusters = [1]
    for i, point in enumerate(unique_points):
        mask = np.where(image[:, :, 0] == point[0], np.ones((H, W)).astype(bool), False) \
                   & np.where(image[:, :, 1] == point[1], np.ones((H, W)).astype(bool), False) \
                   & np.where(image[:, :, 2] == point[2], np.ones((H, W)).astype(bool), False)
        clustered_image = np.where(np.logical_not(mask), clustered_image, clusters[i]+1)
    return clustered_image


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        samples = np.zeros([h * w, 3], dtype=np.float32)
    else:
        samples = np.zeros([h * w], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20,
                                               1.0),
                                              rounds,
                                              cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))