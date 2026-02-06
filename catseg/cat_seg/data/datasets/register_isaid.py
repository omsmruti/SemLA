import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap

CLASSES = (
    "background",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "bridge",
    "large_vehicle",
    "small_vehicle",
    "helicopter",
    "swimming_pool",
    "roundabout",
    "soccer_ball_field",
    "plane",
    "harbor",
)


CLASSES_OFFICIAL = (
    "background",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "bridge",
    "large_vehicle",
    "small_vehicle",
    "helicopter",
    "swimming_pool",
    "roundabout",
    "soccer_ball_field",
    "plane",
    "harbor",
)

# isprs_potsdam                             isprs_potsdam_sem_seg_test_rgb
def register_dataset(root):
    ds_name = 'isaid'
    root = os.path.join(root, 'isaid')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('train', 'train/images/', 'train/annotations_detectron2', CLASSES),
        ('val', 'val/images', 'val/annotations_detectron2', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='png'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True)[[3, 1, 4, 15, 2, 0]],
            classes_of_interest=list(range(len(class_names) - 1)),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
