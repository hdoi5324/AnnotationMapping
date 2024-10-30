import os
import shutil


class SourceData:
    def __init__(self):
        self.test_split = 0.3
        self.split_dict = {}
        self.image_id = 0
        self.ann_id = 0

    def _get_image(self, split, no):
        return self.split_dict[split][no]

    def copy_image_to_split(self, split, orig_path, new_filename, replace=False):
        # Copy image to split directory (test or train)
        subdir = self.subdir_paths[f"{split}{self.opt.year}"]
        new_path = os.path.join(subdir, new_filename)
        if not os.path.exists(new_path) or replace:
            shutil.copy(orig_path, new_path)
        return subdir

