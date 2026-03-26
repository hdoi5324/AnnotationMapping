import os

from tqdm import tqdm

from .coco_writer import CocoWriter
from .sinks import write_box_txt, write_viz


def generate_coco_annotations(dataset, annotation_path, coco_mapping_list):
    # Process each split of images eg train, test
    for split, split_list in dataset.split_dict.items():
        dataset_file = os.path.join(annotation_path,
                                    f"instances_{split}.json")
        writer = CocoWriter(coco_mapping_list, id_policy="sequential")
        if os.path.exists(dataset_file):
            writer.load_existing(dataset_file)

        # Process image list
        for i in tqdm(range(len(split_list))):
            # Get image data
            img_data, image_annotations, img_path = dataset.get_image_data(split, i)
            coco_image_id = writer.add_image(img_data)
            writer.add_annotations(coco_image_id, image_annotations)

            # Optional sidecar outputs (default off; opt-in via Hydra config)
            opt = getattr(dataset, "opt", None)
            root_opt = getattr(dataset, "root_opt", None)
            write_txt = bool(getattr(opt, "write_box_txt", False)) if opt is not None else False
            write_img_viz = bool(getattr(opt, "write_viz", False)) if opt is not None else False
            if root_opt is not None:
                write_txt = write_txt or bool(getattr(root_opt, "write_box_txt", False))
                write_img_viz = write_img_viz or bool(getattr(root_opt, "write_viz", False))

            if image_annotations and write_txt:
                write_box_txt(img_path, image_annotations)
            if image_annotations and write_img_viz:
                write_viz(img_path, image_annotations)

        writer.dump(dataset_file)
        print(
            f"Saved to {dataset_file} with {len(writer.dataset.get('images', []))} images "
            f"and {len(writer.dataset.get('annotations', []))} annotations"
        )
