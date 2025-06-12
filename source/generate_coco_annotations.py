import os

import hydra
from omegaconf import DictConfig

from coco_mapping.generate_coco_annotations import generate_coco_annotations
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
        from coco_mapping.infinigen import InfinigenData
        dataset = InfinigenData(output_dir, opt=opt.dataset, subdir_paths=subdir_paths)
        # Create dir for no water images
        for subdir in [f"test{opt.dataset.year}_nowater", f"train{opt.dataset.year}_nowater"]:
            p = os.path.join(output_dir, subdir)
            os.makedirs(p, exist_ok=True)   
    elif opt.dataset.datatype == "Squidle":
        from coco_mapping.squidle_data import SquidleData
        dataset = SquidleData(opt.dataset, image_dir=output_dir, subdir_paths=subdir_paths)
    else:
        print(f"Not implemented: {opt.dataset.datatype}")
    generate_coco_annotations(dataset, subdir_paths['annotations'], opt.dataset.coco_mapping)


if __name__ == "__main__":
    main()
