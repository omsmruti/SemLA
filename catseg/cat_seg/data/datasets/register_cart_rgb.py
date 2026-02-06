import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap



CLASSES = ('unknown',
            'background',
            'bare_ground', 
            'boulders_rocky_terrain', 
            'human_made_structures',
            'road',
            'shrubs',
            'trees',
            'sky',
            'water',
            'vehicles',
            'person',
)

    


def register_cartr_rgb(root):
    # ds_name = 'indraeye'
    # ds_name = 'IE_day'
    ds_name = 'cartr'
    # root = os.path.join(root, 'Indraeye/eo')
    root = os.path.join(root, 'CART')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('train', 'train/color', 'train/annotations', CLASSES),
        #('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        # ('test', 'test/images', 'test/annotations', CLASSES),
        ('val', 'val/color', 'val/annotations', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        print(image_dir)
        print(gt_dir)
        #sprint(heyy)
        full_name = f'{ds_name}_sem_seg_{split}'
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
register_cartr_rgb(_root)
 
        
