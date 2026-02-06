import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap

CLASSES = (
    'Unlabelled',
    'Bareland'
    'Rangeland',
    'Developed_space',
    'Road',
    'Tree',
    'Water',
    'Agriculture_land',
    'Building',
)

def openearthmap():
    openearthmap_classes = ["Unlabelled", "Bareland", "Rangeland", "Developed_space", "Road", "Tree", "Water", "Agriculture_land", "Building"]


    ret = {
        "stuff_classes" : openearthmap_classes,
    }
    return ret

def register_dataset(root):
    ds_name = 'openearth'
    root = os.path.join(root, 'OpenEarthMap')
    meta = openearthmap()

    for split, image_dirname, sem_seg_dirname, class_names in [
        #('test_rgb', 'images_detectron2/test/rgb', 'annotations_detectron2/test', CLASSES),
        #('test_irrg', 'images_detectron2/test/irrg', 'annotations_detectron2/test', CLASSES),
        #('test_irrg_official', 'images_detectron2/test/irrg', 'annotations_detectron2/test', CLASSES_OFFICIAL),
        ('train_rgb', 'train/rgb_images', 'train/labels', CLASSES),
        ('val_rgb', 'test/rgb_images', 'test/labels', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split.split("_")[0]}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='tif', image_ext='tif'
            ),
        )
        #MetadataCatalog.get(split).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)
        MetadataCatalog.get(full_name).set(image_root=image_dir, sem_seg_root=gt_dir,
        evaluator_type='sem_seg', ignore_label=255, background_class=0, 
        classes_of_interest=list(range(1, len(class_names))), **meta)
        #ignore_label=255,
        #stuff_classes=class_names,
        #stuff_colors=colormap(rgb=True),
        #classes_of_interest=list(range(1, len(class_names))),
        #background_class=0,
        #)


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)