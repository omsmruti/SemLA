import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap

CLASSES = ('background(and back-hoe)', 
           'bicycle', 
           'bus', 
           'car',
           'cargo_trike',                  
           'ignore',
           'motorcycle',
           'person',
           'rickshaw',
           'small_truck',
           'tractor',
           'truck',
           'van',)


def register_dataset(root):
    #ds_name = 'indraeye'
    ds_name = 'IE_Segmentation'
    # root = os.path.join(root, 'Indraeye/eo')
    root = os.path.join(root, 'indraeye/eo')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('train', 'train/images', 'train/annotations', CLASSES),
        #('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        # ('test', 'test/images', 'test/annotations', CLASSES),
        ('val', 'test/images', 'test/annotations', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        print(image_dir)
        print(gt_dir)
        #sprint(heyy)
        full_name = f'indraeye_sem_seg_{split}'  ##'indraeye_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='jpg'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
