# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom


############### SAM2 ###############
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

from einops import rearrange

# Conv-LoRA MoE loss collection
from peft.tuners.lora import Linear as LoraLinear


def collect_conv_lora_moe_loss(model: nn.Module) -> torch.Tensor:
    """
    Collect and sum MoE losses from all LoRA Linear layers with Conv-LoRA enabled.
    
    This function is called after forward pass to collect the load balancing losses
    from all Conv-LoRA layers, which should be added to the main task loss.
    
    Args:
        model: The model containing LoRA Linear layers with Conv-LoRA.
        
    Returns:
        Total MoE loss summed across all Conv-LoRA enabled layers.
    """
    total_loss = None
    device = None
    dtype = None
    
    for module in model.modules():
        if isinstance(module, LoraLinear):
            moe_loss = getattr(module, '_moe_loss', None)
            if moe_loss is not None and moe_loss.item() != 0.0:
                if device is None:
                    device = moe_loss.device
                    dtype = moe_loss.dtype
                if total_loss is None:
                    total_loss = moe_loss
                else:
                    total_loss = total_loss + moe_loss
    
    if total_loss is None:
        # No Conv-LoRA layers found or no MoE loss (Conv-LoRA not enabled)
        return torch.tensor(0.0, device=device if device else "cuda", dtype=dtype if dtype else torch.float32)
    
    return total_loss

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone

        ######################## SAM2 ########################
        # checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():

            # print(HEY)
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))

    def reset_forward_hooks(self):
        """
        Reset the hooks after LoRAs are attached to resblocks (which disables previously set hooks)
        """
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        

        
        images = [x["image"].to(self.device) for x in batched_inputs]

        # print(batched_inputs, "__-----___----___--")
        # print(OIHO)

        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        # print("_-_CLIP IMAGE INPUT SHAPE..._-__", self.clip_resolution)
        # print(HEYpoj)


        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        # clip_images_resized1 = F.interpolate(clip_images.tensor, scale_factor=1.5, mode='bilinear', align_corners=False, )
        # clip_images_resized2 = F.interpolate(clip_images.tensor, scale_factor=0.5, mode='bilinear', align_corners=False, )

        # print("_-_CLIP IMAGE1 INPUT SHAPE..._-__", clip_images_resized.shape)  #torch.Size([2, 3, 336, 336])
        # print("_-_CLIP IMAGE2 INPUT SHAPE..._-__", clip_images_resized1.shape)  #torch.Size([2, 3, 336, 336])
        # print("_-_CLIP IMAGE3 INPUT SHAPE..._-__", clip_images_resized2.shape)  #torch.Size([2, 3, 336, 336])
        # print(OEHYU)

        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        # The above line will be calling the following file and function: "/home/SemLA/catseg/cat_seg/third_party/model_vpt.py", line 427, in encode_image

    #######################################################################
        image_features = clip_features[:, 1:, :]

        # CLIP ViT features for guidance

        ######## HERE on its the decoder network that upsamples the images..#################
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        # print("---___------____----___--_", res4.shape, "BEFORE R4 UPSAMP__--_____---_____-__")
        # print("---___------____----___--_", res5.shape, "BEFORE R5 UPSAMP__--_____---_____-__")
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        # print("---___------____----___--_", res4.shape, "AFTER R4 UPSAMP__--_____---_____-__")
        # print("---___------____----___--_", res5.shape, "AFTER R5 UPSAMP__--_____---_____-__")
        # print(HIOHOI)
        features = {'res5': res5, 'res4': res4, 'res3': res3,}

        # print("----_____---____---___---____", features["res3"].shape, "Scaled clip features....")
        # print("----_____---____---___---____", features["res4"].shape, "Scaled clip features....")
        # print("----_____---____---___---____", features["res5"].shape, "Scaled clip features....")
        # print(HEY)

        outputs = self.sem_seg_head(clip_features, features)
        # print("__--____----___--____", outputs.shape, "__---____--____---_")  # torch.size(2, 19, 96, 96)
        # print(OHIHIOH)


        ##################################################################################################
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            # print("____-----___---____---___--____", targets.shape, outputs.shape, "__---____---___---____--_____")
            # print("____-----___---____---___--____", targets[0].unsqueeze(0).detach().cpu().shape, outputs[0][0].unsqueeze(0).detach().cpu().shape, "__---____---___---____--_____")
            import cv2
            import time

            # cv2.imwrite("target.png", targets[0].unsqueeze(0).permute(1,2,0).detach().cpu().numpy() )
            # cv2.imwrite("target1.png", targets[1].unsqueeze(0).permute(1,2,0).detach().cpu().numpy() )
            # for i in range(outputs.shape[1]):
            #     cv2.imwrite("pred"+ str(i)+".png", outputs[0][i].unsqueeze(0).permute(1,2,0).detach().cpu().numpy()*255 )
            #     time.sleep(1)
            # cv2.imwrite("pred.png", outputs[0][0].unsqueeze(0).permute(1,2,0).detach().cpu().numpy()*255 )
            # print(OJOIJO)
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot

            # outputs_tmp = outputs[0].unsqueeze(0).permute(1,2,0)
            # target_tmp = _targets[0].unsqueeze(0).permute(1,2,0)

            a = outputs[0].permute(2,0,1)
            b = _targets[0].permute(2,0,1)


            # for i in range(num_classes):
            #     cv2.imwrite("pred"+ str(i)+".png", a[i].detach().cpu().numpy()*255 )
            #     cv2.imwrite("target"+ str(i)+".png", b[i].detach().cpu().numpy()*255 )
            #     time.sleep(1)

            # print("____----___---____--", outputs[0].permute(2,0,1).shape , "____--_____---_____-____")
            # print("_--_____----____", _targets.shape, "__----___----___---__")   
            # Above gives the following: torch.Size([2, 384, 384, 19]) as shape with 19 referring to the num of classes...
            # print(OHO)

            ############################ SAM2 ############################
            # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            #     self.predictor.set_image(images)
            #     masks, _, _ = self.predictor.predict("Identify the following objects: car, person, building, sky and road")

            #     print("_-_____--____---____--_____", masks.shape, "********888*****88*****88888*")
            #     print(HEY)

            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            
            # Register hook to check gradients after backward
            def check_grads_hook(grad):
                print("\n" + "="*80)
                print("Checking sem_seg_head parameters for None/NaN gradients:")
                print("="*80)
                
                none_params = []
                nan_params = []
                
                for name, param in self.sem_seg_head.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            none_params.append(name)
                        elif torch.isnan(param.grad).any():
                            nan_params.append(name)
                
                if none_params:
                    print(f"\nParameters with None gradients ({len(none_params)}):")
                    for name in none_params:
                        print(f"  - {name}")
                        pass
                
                if nan_params:
                    print(f"\nParameters with NaN gradients ({len(nan_params)}):")
                    for name in nan_params:
                        #print(f"  - {name}")
                        pass
                
                if not none_params and not nan_params:
                    print("\n✓ All sem_seg_head parameters have valid gradients")
                
                print("="*80 + "\n")
                return grad
            
            #loss.register_hook(check_grads_hook)

            losses = {"loss_sem_seg" : loss}
            
            # Collect MoE loss from Conv-LoRA layers (if enabled)
            moe_loss = collect_conv_lora_moe_loss(self)
            #print("MOE LOSS:---------------------------------", moe_loss)
            if moe_loss.item() > 0:
                losses["loss_conv_lora_moe"] = moe_loss
            
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]

            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)

        # import cv2
        # # print("=---------", output.shape, "_***888****888***")
        # # print(HOH)
        # for i in range(13):
        #     cv2.imwrite("pred"+ str(i)+".png", output[i].detach().cpu().numpy()*255 )
        # print(HEY)
        return [{'sem_seg': output}]