# AnnotationMapping
Code to generate coco train and test files from two sources - Squidle+ data and infinigen synthetic data.  

The code relies on config files in config/dataset and running the python script `source/generate_coco_annotataions.py`.

The script generates three directories - annotations, train2023, test2023.

The config files are based on [hydra.cc](hydra.cc) configurations which are hierarchical in nature.  Configs include details like test/train split, mappings, exclusions, target folders etc.


### Squidle+
Squidle+ is an online marine image repository found at [squidle.org](squidle.org).

This code will generate COCO annotation files from squidle annotation sets by retrieving images and annotations based on specified annotation set ids.

Requires a login to squidle.org to generate an api_token.  The api tokens can be setup in `config/dataset/squidle_default.yaml`

Use config files in `config/dataset` to define annotation set ids, mappings of squidle data to coco data, test/train split, directory for files generated etc.  Specify the config file in the command as the dataset (`squidle_urchin_defualt.yaml` in command below.)

The code also has capability to generate new annotation sets in squidle but this is more advanced.  

```commandline
python source/generate_coco_annotations.py --config-name=config_coco_squidle.yaml dataset="squidle_hand_default.yaml" dataset.annotation_set_ids=[15800] dataset.folder=squidle_handfish_15800
```

### Infinigen
Generate COCO annotations from infinigen output.  Requires Images, Objects, ObjectSegmentation and InstanceSegmentations folders.

Use the config file `config/dataset/infinigen_default.yaml` to specify source folder, mappings between infinigen object names and coco categories and test/train split. 

The config file `config/config_coco_infinigen.yaml` specifies the directory of infinigen data (output_dir).

```commandline
python source/generate_coco_annotations.py --config-name=config_coco_infinigen.yaml dataset.folder="nudi_handfish_auv_v2"
```


### Todo
Add explanation of each config setting.
