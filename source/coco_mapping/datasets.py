import os
import shutil


class SourceData:
    def __init__(self):
        self.test_split = 0.3
        self.split_dict = {}
        self.image_id = 0
        self.ann_id = 0

    def _get_image(self, split, idx):
        return self.split_dict[split][idx]

    def copy_image_to_split(self, split, orig_path, new_filename, replace=False):
        # Copy image to split directory (test or train)
        subdir = self.subdir_paths[f"{split}{self.opt.year}"]
        new_path = os.path.join(subdir, new_filename)
        if not os.path.exists(new_path) or replace:
            shutil.copy(orig_path, new_path)
        return subdir
    
    def get_image_data(self, split, idx):
        """
        Return image data dict, annotation list, and image path for processing.
        
        :param split: 'test' or 'train'
        :param idx: integer index corresponding to item in split_dict[split] list
        :return: image data dictionary of image data including 'file_name', 'height':, 'id', 'license', 'width'
            : annotation list of dictornaries with 'bbox' [x0, y0, x1, y1 format,, 'category', 'polygon', 'source'
            : path to saved image
        """
        return None

