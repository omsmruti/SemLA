from argparse import Namespace

import logging
import os
from typing import Literal

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")

def get_domain_args(
    domain_name: str,
    split: Literal['train', 'val'],
    mode: str = "lora",
    base_model_path: str = "models/model_final.pth",
    num_gpus: int = 1,
    get_cofing_only: bool = False,
):
    logger_names = ["detectron2", "d2", "fvcore"]
    for name in logger_names:
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()

    parts = domain_name.split("-")

    split_list = ["", "", ""]

    for i, part in enumerate(parts):
        split_list[i] = part

    dataset, domain, sub_domain = split_list

    # Supported configurations
    MODE_CHECK = {"lora"}

    DATASET_CHECK = {
        "cs",
        "acdc",
        "acdc_conv",
        "muses",
        "bdd",
        "mv",
        "a150",
        "idd",
        "pc59",
        "nyu",
        "coconutL",
        "cocostuff",
        "IE_Segmentation",
        "IE_Segmentation_ir",
        "indraeye",
        "indraeyed",
        "indraeyen",
        "msrs_rgb",
        "msrs_ir",
        "cartr",
        "carti",
        "openearth",
        "idd_conv",
        "cs_conv",
        "mv_conv",
        "muses_conv",
        "bdd_conv",
        "uavid",
        "isprs",
        "isaid"
    }

    CS_DOMAIN_CHECK = {"normal", "rain"}
    CS_SUB_DOMAIN_CHECK = ["25mm", "50mm", "75mm", "100mm", "200mm"]

    ACDC_DOMAIN_CHECK = {"fog", "night", "snow", "rain"}

    MUSES_DOMAIN_CHECK = {"clear", "rain", "fog", "snow"}
    MUSES_SUB_DOMAIN_CHECK = {"day", "night"}
    INDRAEYE = {"day", "night"}

    # Configurations assertions
    # assert dataset in DATASET_CHECK

    if dataset == "cs":
        assert (
            domain in CS_DOMAIN_CHECK
        ), "Domain '{domain}' not supported for Cityscapes"
        if domain == "rain":
            assert (
                sub_domain in CS_SUB_DOMAIN_CHECK
            ), "Given volume '{volume}' is not supported for Cityscapes"
        elif domain == "normal":
            assert (
                sub_domain == ""
            ), f"Volume '{sub_domain}' is not supported for this domain in Cityscapes"
    elif dataset == "muses":
        assert (
            domain in MUSES_DOMAIN_CHECK
        ), f"Domain '{domain}' not supported for MUSES"
        assert (
            sub_domain in MUSES_SUB_DOMAIN_CHECK
        ), f"Given illumination '{sub_domain}' is not supported for MUSES"
    elif dataset == "acdc":
        assert (
            domain in ACDC_DOMAIN_CHECK
        ), f"Domain '{domain}' is not supported for ACDC"
        assert sub_domain == "", "Volume is not supported in ACDC"

    assert mode in MODE_CHECK, "Mode '{mode}' not supported"

    configs = {
        "cs": {
            "rain": f"configs/cityscapes/rain/{sub_domain}/{mode}-{domain}-{sub_domain}.yaml",
            "normal": f"configs/cityscapes/normal/{mode}-{domain}.yaml",
        },
        "cs_conv": {
            "rain": f"configs/cityscapes/rain/{sub_domain}/{mode}-{domain}-{sub_domain}.yaml",
            "normal": f"configs/cityscapes/normal/{mode}-{domain}.yaml",
        },
        "acdc": {f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc.yaml"},
        "acdc_conv": {f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc.yaml"},
        "muses": {
            f"{domain}": f"configs/muses/{domain}/muses-{domain}-{sub_domain}.yaml"
        },
        "muses_conv": {
            f"{domain}": f"configs/muses/{domain}/muses-{domain}-{sub_domain}.yaml"
        },
        "bdd": "configs/bdd/bdd.yaml",
        "bdd_conv": "configs/bdd/bdd.yaml",
        "mv": "configs/mv/mv.yaml",
        "mv_conv": "configs/mv/mv.yaml",
        "nyu": "configs/nyu/nyu.yaml",
        "a150": "configs/a150/a150.yaml",
        "idd": "configs/idd/idd.yaml",
        "idd_conv": "configs/idd/idd.yaml",
        'pc59': 'configs/pc59/pc59.yaml',
        'nyu': 'configs/nyu/nyu.yaml',
        'coconutL': 'configs/coconutL/coconutL.yaml',
        "cocostuff": "configs/coco/coco-stuff.yaml",
        "IE_Segmentation": "configs/indraeye/rgb/indraeye-rgb.yaml",
        "indraeye": "configs/indraeye/rgb/indraeye-rgb.yaml",
        "indraeyed": "configs/indraeye/rgb/indraeye-rgb.yaml",
        "indraeyen": "configs/indraeye/rgb/indraeye-rgb.yaml",
        
        # "indraeye": {
        #     "day": f"configs/indraeye/{domain}/{sub_domain}.yaml",
        #     "night": f"configs/indraeye/{domain}/{sub_domain}.yaml",
        # },
        "msrs_rgb": "configs/msrs/rgb/msrs-rgb.yaml",
        "msrs_ir": "configs/msrs/ir/msrs-ir.yaml",
        "cartr": "configs/cart/rgb/cart-rgb.yaml",
        "carti": "configs/cart/ir/cart-ir.yaml",
        "IE_Segmentation_ir": "configs/indraeye/ir/indraeye-ir.yaml",
        "openearth": "configs/openearth/openearth.yaml",
        "uavid": "configs/uavid/uavid.yaml",
        "isprs": "configs/ISPRS_Potsdam/ISPRS_Potsdam.yaml",
        "isaid": "configs/isaid/isaid.yaml",
    }

    datasets = {
        "cs": {
            "normal": {
                "train": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/train/",
                "val": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/val/",
            },
        },
        "cs_conv": {
            "normal": {
                "train": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/train/",
                "val": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/val/",
            },
        },
        "acdc": {
            "train": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/train/",
            "val": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/val/",
        },
        "acdc_conv": {
            "train": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/train/",
            "val": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/val/",
        },
        "muses": {
            "train": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/train/{domain}/{sub_domain}/",
            "val": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/val/{domain}/{sub_domain}/",
        },
        "muses_conv": {
            "train": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/train/{domain}/{sub_domain}/",
            "val": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/val/{domain}/{sub_domain}/",
        },
        "bdd": {
            "train": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/train/",
            "val": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/val/",
        },
        "bdd_conv": {
            "train": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/train/",
            "val": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/val/",
        },
        "mv": {
            "train": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/train/images/",
            "val": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/val/images/",
        },
        "mv_conv": {
            "train": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/train/images/",
            "val": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/val/images/",
        },
        "a150": {
            "train": f"{DETECTRON2_DATASET_PATH}ADE20k/images/training/",
            "val": f"{DETECTRON2_DATASET_PATH}ADE20k/images/validation/",
        },
        "idd": {
            "train": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/val/",
        },
        "pc59": {
            "train": f"{DETECTRON2_DATASET_PATH}pascal_ctx_d2/images/training",
            "val": f"{DETECTRON2_DATASET_PATH}pascal_ctx_d2/images/validation",
        },
        "nyu": {
            "train": f"{DETECTRON2_DATASET_PATH}nyudv2_splitted/train/rgb",
            "val": f"{DETECTRON2_DATASET_PATH}nyudv2_splitted/test/rgb",
        },
        "coconutL": {
            "train": f"{DETECTRON2_DATASET_PATH}coconut-l/train2017/",
            "val": f"{DETECTRON2_DATASET_PATH}coconut-l/val2017",
        },
        "IE_Segmentation": {
            "train": f"{DETECTRON2_DATASET_PATH}indraeye/eo/train/",
            "val": f"{DETECTRON2_DATASET_PATH}indraeye/eo/test/",
        },
        "IE_Segmentation_ir": {
            "train": f"{DETECTRON2_DATASET_PATH}IE_Segmentation/IE_eo_ir_split/ir/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IE_Segmentation/IE_eo_ir_split/ir/val/",
        },
        "cocostuff": {
            "train": f"{DETECTRON2_DATASET_PATH}/coco/train2017/",
            "val": f"{DETECTRON2_DATASET_PATH}/coco/val2017",
        },
        "indraeyed": {
            "train": f"{DETECTRON2_DATASET_PATH}IE_daynight/IE_eo_ir_split/eo/rgbday/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IE_daynight/IE_eo_ir_split/eo/rgbday/val/",
        },
        "indraeye": {
            "train": f"{DETECTRON2_DATASET_PATH}indraeye/eo/train/",
            "val": f"{DETECTRON2_DATASET_PATH}indraeye/eo/test/",
        },
        "indraeyen": {
            "train": f"{DETECTRON2_DATASET_PATH}IE_daynight/IE_eo_ir_split/eo/rgbnight/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IE_daynight/IE_eo_ir_split/eo/rgbnight/val/",
        },
        "msrs_rgb": {
            "train": f"{DETECTRON2_DATASET_PATH}msrs/images/train",
            "val": f"{DETECTRON2_DATASET_PATH}msrs/images/test",
        },
        "msrs_ir": {
            "train": f"{DETECTRON2_DATASET_PATH}msrs/train/ir/",
            "val": f"{DETECTRON2_DATASET_PATH}msrs/test/ir/rgbnight/val/",
        },
        "cartr": {
            "train": f"{DETECTRON2_DATASET_PATH}CART/train/",
            "val": f"{DETECTRON2_DATASET_PATH}CART/val/",
        },
        "carti": {
            "train": f"{DETECTRON2_DATASET_PATH}cart/train/",
            "val": f"{DETECTRON2_DATASET_PATH}cart/val/",
        },
        "openearth": {
            "train": f"{DETECTRON2_DATASET_PATH}OpenEarthMap/train/",
            "val": f"{DETECTRON2_DATASET_PATH}OpenEarthMap/val/",
        },
        "idd_conv": {
            "train": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/val/",
        },
        "uavid": {
            "train": f"{DETECTRON2_DATASET_PATH}uavid_v1.5/images_detectron2/val/",
            "val": f"{DETECTRON2_DATASET_PATH}uavid_v1.5/images_detectron2/val/",
        },
        "isprs": {
            "train": f"{DETECTRON2_DATASET_PATH}ISPRS_Potsdam/images_detectron2/test/rgb/",
            "val": f"{DETECTRON2_DATASET_PATH}ISPRS_Potsdam/images_detectron2/test/rgb/",
        },
        "isaid": {
            "train": f"{DETECTRON2_DATASET_PATH}isaid/train/images/",
            "val": f"{DETECTRON2_DATASET_PATH}isaid/val/images/",
        },
    }

    # Output path configuration
    output_path = (
        f"output/{dataset}/{mode}-{dataset}"
        + (f"-{domain}" if  domain != "" else "")
        + (f"-{sub_domain}" if sub_domain != "" else "")
        + "/eval/"
    )
    

    # print("-___---_____--_____--________", configs,"----___", datasets, "--_____---___----__", domain, "____----___----____-_____--_____")
    # print(OIJ)

    # Constructing the return values
    if domain == "" and sub_domain == "":
        config_file = configs[dataset]
    else:

        # print("-___--______", dataset, domain, "_--___-----___-")
        config_file = configs[dataset][domain]
        # config_file = configs[dataset]

    

    # if dataset == "cs_conv":
    #     train_dataset_path = (
    #         datasets[dataset]["normal"]["train"]
    #         if dataset != "cs"
    #         else datasets[dataset][domain]["train"]
    #     )

    #     val_dataset_path = (
    #         datasets[dataset]["normal"]["val"]
    #         if dataset != "cs"
    #         else datasets[dataset][domain]["val"]
    #     )
    # else:
    #     # import ipdb
    #     # ipdb.set_trace(context=10)
    #     train_dataset_path = (
    #     datasets[dataset]["train"]
    #     if dataset != "cs"
    #     else datasets[dataset][domain]["train"]
    # )

    # val_dataset_path = (
    #     datasets[dataset]["val"]
    #     if dataset != "cs"
    #     else datasets[dataset][domain]["val"]
    # )

    # args = Namespace(
    #     config_file=config_file,
    #     eval_only=True,
    #     num_gpus=num_gpus,
    #     train_dataset_path=train_dataset_path,
    #     val_dataset_path=val_dataset_path,
    #     opts=[
    #         "OUTPUT_DIR",
    #         output_path,
    #         "TEST.SLIDING_WINDOW",
    #         "True",
    #         "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
    #         "[1,1]",
    #         "MODEL.WEIGHTS",
    #         base_model_path,
    #     ],
    #     resume=True,
    # )

    if dataset in ["cs", "cs_conv"]:
        train_dataset_path = datasets[dataset]["normal"]["train"]
        val_dataset_path = datasets[dataset]["normal"]["val"]
    else:
        train_dataset_path = datasets[dataset]["train"]
        val_dataset_path = datasets[dataset]["val"]

    args = Namespace(
        config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        opts=[
            "OUTPUT_DIR",
            output_path,
            "TEST.SLIDING_WINDOW",
            "True",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
            "MODEL.WEIGHTS",
            base_model_path,
        ],
        resume=True,
    )

    if get_cofing_only:
        return args
    else:
        from catseg.train_net import Trainer, setup
        dataset_name = f"{domain_name}_sem_seg_{split}"
        if dataset == "IE_Segmentation":
            dataset_name = f"indraeye_sem_seg_{split}"
        if dataset == "isprs":
            dataset_name = f"isprs_potsdam_sem_seg_{split}"
        data_loader = Trainer.build_test_loader(
            setup(args), dataset_name
        )

        evaluator = Trainer.build_evaluator(setup(args), dataset_name)

        return args, evaluator, data_loader


def custom_domain_args(
    config_file,
    output_path,
    num_gpus=1,
    model_path="models/model_final.pth",
    dataset_path: str = None,
    seed=None,
):

    args = Namespace(
        config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        dataset_path=dataset_path,
        opts=[
            "OUTPUT_DIR",
            output_path,
            "TEST.SLIDING_WINDOW",
            "True",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
            "MODEL.WEIGHTS",
            model_path,
        ],
        resume=True,
        model_path=model_path,
    )

    if seed != None:
        args.opts.extend(["SEED", seed])

    return args


def benchmark_catseg(model, args):

    import detectron2.utils.comm as comm
    from detectron2.evaluation import verify_results

    from catseg.train_net import Trainer, set_random_seed, setup

    cfg = setup(args)
    set_random_seed(cfg.SEED)
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res

def load_catseg_model(args, model_path: str = None):
    from catseg.train_net import Trainer, setup
    from detectron2.checkpoint import DetectionCheckpointer

    print("Loading base model ...")
    
    try:
        cfg = setup(args)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if model_path is None else model_path, resume=args.resume
        )
        print("Base model loaded.\n")
        return model
    except AttributeError as e:
        print(f"Error: Invalid model configuration: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: Model weights not found at the specified path.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading the base model: {e}")
        raise