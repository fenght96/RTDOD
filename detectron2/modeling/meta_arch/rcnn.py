# Copyright (c) Facebook, Inc. and its affiliates.
from email.mime import image
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import copy
import torch.nn.functional as F
from detectron2.structures.boxes import Boxes
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import json
from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork", "GeneralizedRCNNMask", "GeneralizedR", 'GeneralizedRCNNCOD', 'GeneralizedRCNNDOD', 'GeneralizedRCNNDIS']

# for first stage of gendis
@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)[0][0]

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)[0][0]

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        # import pdb; pdb.set_trace()

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results



# for fisrt stage of cod
@META_ARCH_REGISTRY.register()
class GeneralizedRCNNMask(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        task: int=0,
        save_path = None,
        load_path = None,
        load_model_path = None,
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # for n,p in self.backbone.named_parameters():
        #     if 'cls_fcs' not in n:
        #         p.requires_grad = False
        # Task
        self.task = task
        self.dis_flag = False
        embed_features = 512
        self.save_path = save_path
        self.pre_masks = None
        if load_path is not None:
            self.pre_masks = self.load_task_masks(load_path)



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        return {
                "backbone": backbone,
                "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
                "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
                "input_format": cfg.INPUT.FORMAT,
                "vis_period": cfg.VIS_PERIOD,
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "task" : cfg.MODEL.TASK,
                "save_path" : cfg.MODEL.SAVE_PATH,
                "load_path" : cfg.MODEL.LOAD_PATH,
                "load_model_path" : cfg.MODEL.WEIGHTS,
            }
        

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


        

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], detected_instances = None, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        
        # Feature Extraction.
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            features, _, task_masks = self.backbone(images.tensor, self.pre_masks)# src :[now_feat, [old_mask_feat0, old_mask_feat1..] ]
            features = features[0]
            # Prepare Proposals.
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
            # Prediction.
            _, detector_losses, gate,_ = self.roi_heads(images, features, proposals, gt_instances)
            # proposals, losses, gate, pre_predictions        
            task_masks += gate
            loss_dict = {}
            loss_dict.update(detector_losses)
            loss_dict.update(proposal_losses)
            self.save_task_masks(task_masks, self.pre_masks)
            return loss_dict

        else:
            pre_masks = self.load_task_masks(self.save_path)
            features, _, _ = self.backbone(images.tensor, pre_masks)# for testing:  [dict(list(zip(self._out_features, results))), None], 0, None
            # Prediction.
            features = features[0]
            # Prepare Proposals.
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            results, _  = self.roi_heads(images, features, proposals)
            # import pdb; pdb.set_trace()
            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            else:
                return results

        
    def save_task_masks(self, task_masks, pre_masks=None):
        save_masks = [[x[0].detach().cpu().numpy().tolist(),] for x in task_masks]
        
        if pre_masks is not None:
            tmp = []
            for mak, pre in zip(save_masks, pre_masks):
                tmp.append(mak + [p.detach().cpu().numpy().tolist() for p in pre])
            save_masks = tmp
        save_masks = {'{}'.format(i):v for i, v in enumerate(save_masks) }
        
        
        jsObj = json.dumps(save_masks)  
  
        fileObject = open(self.save_path, 'w')  
        fileObject.write(jsObj)  
        fileObject.close()
        
    def load_task_masks(self, load_path):
        with open(load_path,'r', encoding='UTF-8') as f:
            load_dict = json.load(f)
        task_masks = []
        for k,v in load_dict.items():
            task_masks.append([torch.from_numpy(np.array(m)).cuda().float() for m in v])
        return task_masks




    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        return self.forward(batched_inputs, detected_instances, do_postprocess)

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

# for class
@META_ARCH_REGISTRY.register()
class GeneralizedRCNNCOD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        task: int=0,
        save_path = None,
        load_path = None,
        load_model_path = None,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_backbone= None,
        dis_proposal_generator= None,
        dis_roi_heads= None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # for n,p in self.backbone.named_parameters():
        #     if 'cls_fcs' not in n:
        #         p.requires_grad = False
        # Task
        self.task = task
        self.dis_flag = False
        embed_features = 512
        self.dis_gen = False
        self.save_path = save_path
        self.pre_masks = None
        if load_path is not None:
            self.pre_masks = self.load_task_masks(load_path)



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        assert cfg.MODEL.TASK > 0 
            
        dis_backbone = build_backbone(cfg)
        return {
                "backbone": backbone,
                "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
                "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
                "input_format": cfg.INPUT.FORMAT,
                "vis_period": cfg.VIS_PERIOD,
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "task" : cfg.MODEL.TASK,
                "save_path" : cfg.MODEL.SAVE_PATH,
                "load_path" : cfg.MODEL.LOAD_PATH,
                "load_model_path" : cfg.MODEL.WEIGHTS,
                "dis_backbone": dis_backbone,
                "dis_proposal_generator": build_proposal_generator(cfg, dis_backbone.output_shape()),
                "dis_roi_heads": build_roi_heads(cfg, dis_backbone.output_shape()),
            }


    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    def dis_gen_model(self):
        # for p in self.roi_heads.parameters():
        #     p.requires_grad = False
        self.dis_backbone = copy.deepcopy(self.backbone)
        self.dis_proposal_generator = copy.deepcopy(self.proposal_generator)
        self.dis_roi_heads = copy.deepcopy(self.roi_heads)
        for p in self.dis_backbone.parameters():
                p.requires_grad = False
        for p in self.dis_roi_heads.parameters():
            p.requires_grad = False
        for p in self.dis_proposal_generator.parameters():
            p.requires_grad = False
        self.dis_gen = True

        

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], detected_instances = None, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images = self.preprocess_image(batched_inputs)
        # if isinstance(images, (list, torch.Tensor)):
        #     images = nested_tensor_from_tensor_list(images)
        
        
        
        # Feature Extraction.
        if self.training:
            if self.dis_gen is False and self.task > 0:
                self.dis_gen_model()
            if self.task > 0 and self.dis_gen:
                self.dis_roi_heads.eval()
                self.dis_backbone.eval()
                self.dis_proposal_generator.eval()
                # self.roi_heads.eval()
                
            dis_loss = []
            old_features, old_backbone_features, old_labels = self.old_inference(images, self.pre_masks, batched_inputs)
            old_labels = old_labels[0][0]
            gt_merges = []
            gt_olds = []
            for i in range(len(batched_inputs)):
                gt_merge, gt_old = self.merge_result(old_labels[i], batched_inputs[i]["instances"])
                gt_merges.append(gt_merge.to(self.device))
                gt_olds.append(gt_old.to(self.device))
            gt_news = [x['instances'].to(self.device) for x in batched_inputs]
            # gt_olds = [x['instances'].to(self.device) for x in batched_inputs]
            # gt_merges = [x['instances'].to(self.device) for x in batched_inputs]
            
            features, backbone_features, task_masks = self.backbone(images.tensor, self.pre_masks)# src :[now_feat, [old_mask_feat0, old_mask_feat1..] ]
            
            pre_task_flag = len(features) >= 2
            if pre_task_flag:
                pre_feats = features[1:]
                for sr, dis_sr in zip(pre_feats, old_features):
                    for k,v in sr.items():
                        dis_loss.append(F.l1_loss(sr[k], dis_sr[k].detach()))
                for k,vs in backbone_features.items():
                    for i in range(1, len(vs)):
                        dis_loss.append(F.l1_loss(vs[i], old_backbone_features[k][i-1].detach()))
            loss_dict = {}
            for i, feature in enumerate(features): # i for task index
                # Prepare Proposals.
                if self.proposal_generator is not None:
                    proposals, proposal_losses = self.proposal_generator(images, feature, gt_merge)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                if i == 0:
                    _, detector_losses, gates, _ = self.roi_heads(images, feature, proposals, gt_news)
                    task_masks.append([gates,])
                    for k,v in detector_losses.items():
                        if k not in loss_dict.keys():
                            loss_dict[k] = v
                        else:
                            loss_dict[k] += v
                    # print(f'i:{i}', loss_dict)
                elif len(gt_olds[0]) > 0:
                    _, detector_losses, _, _ = self.roi_heads(images, feature, proposals, gt_olds, self.pre_masks[-1][i-1]) # -1 for last stage and i for task index
                    for k,v in detector_losses.items():
                        if k not in loss_dict.keys():
                            loss_dict[k] = v
                        else:
                            loss_dict[k] += v
                    # print(detector_losses)
                # print(f'i:{i}', loss_dict)
                
            loss_dict.update(proposal_losses)
            tmp = sum(dis_loss) / len(dis_loss) * 0.5
            if not torch.isinf(tmp):
                loss_dict['loss_dis'] = tmp
            
            self.save_task_masks(task_masks, self.pre_masks)
            return loss_dict

        else:
            pre_masks = self.load_task_masks(self.save_path)
            features, _, _ = self.backbone(images.tensor, pre_masks)# for testing:  [dict(list(zip(self._out_features, results))), None], 0, None
            # Prediction.
            results = []            
            for i, feature in enumerate(features):
                # Prepare Proposals.
                if self.proposal_generator is not None:
                    proposals, _ = self.proposal_generator(images, feature, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                for j in range(len(pre_masks[-1][i])):
                    result = self.roi_heads(images, feature, proposals, None, pre_masks[-1][i][j])
                    results.append(result)
            # import pdb; pdb.set_trace()
            if do_postprocess:
                processed_results = []
                for result in results:# [N for task [bs * outputs]]
                    for results_per_image, input_per_image, image_size in zip(result, batched_inputs, images.image_sizes):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        r = detector_postprocess(results_per_image[0], height, width)
                        processed_results.append({"instances": r})
                # multi instances to one instance
                processed_results = self.merge_results(processed_results)
                return processed_results
            else:
                return results
    
    def merge_results(self, results):
        saved_boxes = None
        saved_cls = None
        saved_scores = None
        
        for result in results:
            result = result['instances']
            if len(result) <= 0:
                continue
            scores = result.get('scores')
            pred_boxes = result.get('pred_boxes').tensor
            pred_classes = result.get('pred_classes')
            if saved_boxes is None:
                saved_boxes = pred_boxes
                saved_cls = pred_classes
                saved_scores = scores
            else:
                saved_boxes = torch.cat([saved_boxes, pred_boxes],dim=0)
                saved_cls = torch.cat([saved_cls, pred_classes],dim=0)
                saved_scores = torch.cat([saved_scores, scores],dim=0)
        if isinstance(saved_boxes, torch.Tensor):
            pred = Instances(results[0]['instances']._image_size, **{'pred_boxes': Boxes(saved_boxes), 'pred_classes': saved_cls, 'scores':saved_scores}) 
        else:
            pred = Instances(results[0]['instances']._image_size, **{'pred_boxes': Boxes(torch.tensor([]).to(self.device)), 'pred_classes': torch.tensor([]).to(self.device), "scores": torch.tensor([])}) 
        return [{'instances':pred},]

    def merge_result(self, results, gt_results):
        result = results
        tmp = copy.deepcopy(gt_results) # fields = 'gt_boxes' 'gt_classes'
        scores = result.get('scores').cpu()
        pred_boxes = result.get('pred_boxes').tensor.cpu()
        pred_classes = result.get('pred_classes').cpu()
        gt_boxes = tmp.get('gt_boxes').tensor.cpu()
        gt_classes = tmp.get('gt_classes').cpu()
        saved_boxes = None
        saved_cls = None
        # print(pred_boxes)
        for i, score in enumerate(scores):
            if score >= 0.8:
                gt_boxes = torch.cat([gt_boxes, pred_boxes[i].unsqueeze(0)],dim=0)
                gt_classes = torch.cat([gt_classes, pred_classes[i].unsqueeze(0)],dim=0)
                if saved_boxes is None:
                    saved_boxes = pred_boxes[i].unsqueeze(0)
                    saved_cls = pred_classes[i].unsqueeze(0)
                else:
                    saved_boxes = torch.cat([saved_boxes, pred_boxes[i].unsqueeze(0)],dim=0)
                    saved_cls = torch.cat([saved_cls, pred_classes[i].unsqueeze(0)],dim=0)
        
        tmp = Instances(gt_results._image_size, **{'gt_boxes': Boxes(gt_boxes), 'gt_classes': gt_classes})
        if isinstance(saved_boxes, torch.Tensor):
            pred = Instances(gt_results._image_size, **{'gt_boxes': Boxes(saved_boxes), 'gt_classes': saved_cls}) 
        else:
            pred = Instances(gt_results._image_size, **{'gt_boxes': Boxes(torch.tensor([])), 'gt_classes': torch.tensor([])}) 
        return tmp, pred
        

    def old_inference(self, images, pre_masks, batched_inputs):
        features, bottom_up_features, _ = self.dis_backbone(images.tensor, pre_masks)
        results = [] 
        for i, feature in enumerate(features): # i for task_id
            proposals, _ = self.dis_proposal_generator(images, feature, None)
            for j in range(len(pre_masks[-1][i])):
                result = self.dis_roi_heads(images, feature, proposals, None, pre_masks[-1][i][j])
                results.append(result)
        # multi instances to one instance
        return features, bottom_up_features, results
        # return features, results

        
    def save_task_masks(self, task_masks, pre_masks=None):
        save_masks = [[x[0].detach().cpu().numpy().tolist(),] for x in task_masks]
        
        if pre_masks is not None:
            tmp = []
            for mak, pre in zip(save_masks, pre_masks):
                tmp.append(mak + [p.detach().cpu().numpy().tolist() for p in pre])
            save_masks = tmp
        save_masks = {'{}'.format(i):v for i, v in enumerate(save_masks) }
        
        
        jsObj = json.dumps(save_masks)  
  
        fileObject = open(self.save_path, 'w')  
        fileObject.write(jsObj)  
        fileObject.close()
        
    def load_task_masks(self, load_path):
        with open(load_path,'r', encoding='UTF-8') as f:
            load_dict = json.load(f)
        task_masks = []
        for k,v in load_dict.items():
            task_masks.append([torch.from_numpy(np.array(m)).cuda().float() for m in v])
        return task_masks




    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        return self.forward(batched_inputs, detected_instances, do_postprocess)

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

# for pretrain
@META_ARCH_REGISTRY.register()
class GeneralizedR(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_trm = copy.deepcopy(backbone)
        for k,v in self.backbone.named_parameters():
            v.requires_grad = False

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": None,#build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": None,#build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."

        images = self.preprocess_image(batched_inputs)

        rgb = images.tensor[:,:3,:,:]
        trm = images.tensor[:,3:,:,:]
        labels = self.backbone(rgb)
        preds = self.backbone_trm(trm)

        loss = 0 
        for k,v in labels.items():
            loss = torch.nn.functional.mse_loss(preds[k], labels[k].detach())#(labels[k] - preds[k])**2

        losses = {}
        losses.update({'reg_loss':loss})
        return losses


    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

class GeneralizedRCNNDODMask(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        task: int=0,
        save_path = None,
        load_path = None,
        load_model_path = None,
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # for n,p in self.backbone.named_parameters():
        #     if 'cls_fcs' not in n:
        #         p.requires_grad = False
        # Task
        self.task = task
        self.dis_flag = False
        embed_features = 512
        self.save_path = save_path
        self.pre_masks = None
        if load_path is not None:
            self.pre_masks = self.load_task_masks(load_path)



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        return {
                "backbone": backbone,
                "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
                "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
                "input_format": cfg.INPUT.FORMAT,
                "vis_period": cfg.VIS_PERIOD,
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "task" : cfg.MODEL.TASK,
                "save_path" : cfg.MODEL.SAVE_PATH,
                "load_path" : cfg.MODEL.LOAD_PATH,
                "load_model_path" : cfg.MODEL.WEIGHTS,
            }
        

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


        

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], detected_instances = None, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        
        # Feature Extraction.
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            features, _, task_masks = self.backbone(images.tensor, self.pre_masks)# src :[now_feat, [old_mask_feat0, old_mask_feat1..] ]
            features = features[0]
            # Prepare Proposals.
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
            # Prediction.
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
            # proposals, losses, gate, pre_predictions        
                
            loss_dict = {}
            loss_dict.update(detector_losses)
            loss_dict.update(proposal_losses)
            self.save_task_masks(task_masks, self.pre_masks)
            return loss_dict

        else:
            pre_masks = self.load_task_masks(self.save_path)
            features, _, _ = self.backbone(images.tensor, pre_masks)# for testing:  [dict(list(zip(self._out_features, results))), None], 0, None
            # Prediction.
            features = features[0]
            # Prepare Proposals.
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            results, _  = self.roi_heads(images, features, proposals)
            # import pdb; pdb.set_trace()
            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            else:
                return results

        
    def save_task_masks(self, task_masks, pre_masks=None):
        save_masks = [[x[0].detach().cpu().numpy().tolist(),] for x in task_masks]
        
        if pre_masks is not None:
            tmp = []
            for mak, pre in zip(save_masks, pre_masks):
                tmp.append(mak + [p.detach().cpu().numpy().tolist() for p in pre])
            save_masks = tmp
        save_masks = {'{}'.format(i):v for i, v in enumerate(save_masks) }
        
        
        jsObj = json.dumps(save_masks)  
  
        fileObject = open(self.save_path, 'w')  
        fileObject.write(jsObj)  
        fileObject.close()
        
    def load_task_masks(self, load_path):
        with open(load_path,'r', encoding='UTF-8') as f:
            load_dict = json.load(f)
        task_masks = []
        for k,v in load_dict.items():
            task_masks.append([torch.from_numpy(np.array(m)).cuda().float() for m in v])
        return task_masks




    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        return self.forward(batched_inputs, detected_instances, do_postprocess)

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results



# for domain
@META_ARCH_REGISTRY.register()
class GeneralizedRCNNDOD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        task: int=0,
        save_path = None,
        load_path = None,
        load_model_path = None,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_backbone= None,
        dis_proposal_generator= None,
        dis_roi_heads= None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # for n,p in self.backbone.named_parameters():
        #     if 'cls_fcs' not in n:
        #         p.requires_grad = False
        # Task
        self.task = task
        self.dis_flag = False
        embed_features = 512
        self.dis_gen = False
        self.save_path = save_path
        self.pre_masks = None
        if load_path is not None:
            self.pre_masks = self.load_task_masks(load_path)



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        assert cfg.MODEL.TASK > 0
        
        dis_backbone = build_backbone(cfg)
        return {
                "backbone": backbone,
                "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
                "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
                "input_format": cfg.INPUT.FORMAT,
                "vis_period": cfg.VIS_PERIOD,
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "task" : cfg.MODEL.TASK,
                "save_path" : cfg.MODEL.SAVE_PATH,
                "load_path" : cfg.MODEL.LOAD_PATH,
                "load_model_path" : cfg.MODEL.WEIGHTS,
                "dis_backbone": dis_backbone,
                "dis_proposal_generator": build_proposal_generator(cfg, dis_backbone.output_shape()),
                "dis_roi_heads": build_roi_heads(cfg, dis_backbone.output_shape()),
            }


    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    def dis_gen_model(self):
        # for p in self.roi_heads.parameters():
        #     p.requires_grad = False
        self.dis_backbone = copy.deepcopy(self.backbone)
        self.dis_proposal_generator = copy.deepcopy(self.proposal_generator)
        self.dis_roi_heads = copy.deepcopy(self.roi_heads)
        for p in self.dis_backbone.parameters():
                p.requires_grad = False
        for p in self.dis_roi_heads.parameters():
            p.requires_grad = False
        for p in self.dis_proposal_generator.parameters():
            p.requires_grad = False
        self.dis_gen = True

        

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], detected_instances = None, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images = self.preprocess_image(batched_inputs)
        # if isinstance(images, (list, torch.Tensor)):
        #     images = nested_tensor_from_tensor_list(images)
        
        
        
        # Feature Extraction.
        if self.training:
            if self.dis_gen is False:
                self.dis_gen_model()
            if self.task > 0 and self.dis_gen:
                self.dis_roi_heads.eval()
                self.dis_backbone.eval()
                self.dis_proposal_generator.eval()
                # self.roi_heads.eval()
                
            dis_loss = []
            old_features, old_backbone_features, _ = self.old_inference(images, self.pre_masks, batched_inputs)
            # old_labels = old_labels[0][0]
            # gt_merges = []
            # gt_olds = []
            # for i in range(len(batched_inputs)):
            #     gt_merge, gt_old = self.merge_result(old_labels[i], batched_inputs[i]["instances"])
            #     gt_merges.append(gt_merge.to(self.device))
            #     gt_olds.append(gt_old.to(self.device))
            gt_news = [x['instances'].to(self.device) for x in batched_inputs]
            # gt_olds = [x['instances'].to(self.device) for x in batched_inputs]
            # gt_merges = [x['instances'].to(self.device) for x in batched_inputs]
            
            features, backbone_features, task_masks = self.backbone(images.tensor, self.pre_masks)# src :[now_feat, [old_mask_feat0, old_mask_feat1..] ]
            
            pre_task_flag = self.task > 0
            if pre_task_flag:
                if len(features) > 0:
                    pre_feats = features[1:]
                    for sr, dis_sr in zip(pre_feats, old_features):
                        for k,v in sr.items():
                            dis_loss.append(F.l1_loss(sr[k], dis_sr[k].detach()))
                for k,vs in backbone_features.items():
                    for i in range(1, len(vs)):
                        dis_loss.append(F.l1_loss(vs[i], old_backbone_features[k][i-1].detach()))
            loss_dict = {}
            for i, feature in enumerate(features): # i for task index
                # Prepare Proposals.
                if self.proposal_generator is not None:
                    proposals, proposal_losses = self.proposal_generator(images, feature, gt_news)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                if i == 0:
                    _, detector_losses = self.roi_heads(images, feature, proposals, gt_news)
                    for k,v in detector_losses.items():
                        if k not in loss_dict.keys():
                            loss_dict[k] = v
                        else:
                            loss_dict[k] += v
                    # print(f'i:{i}', loss_dict)
                else:
                    _, detector_losses = self.roi_heads(images, feature, proposals, gt_news) # -1 for last stage and i for task index
                    for k,v in detector_losses.items():
                        if k not in loss_dict.keys():
                            loss_dict[k] = v
                        else:
                            loss_dict[k] += v
                    # print(detector_losses)
                # print(f'i:{i}', loss_dict)
                
            loss_dict.update(proposal_losses)
            tmp = sum(dis_loss) / len(dis_loss) #* 0.5
            if not torch.isinf(tmp):
                loss_dict['loss_dis'] = tmp
            
            self.save_task_masks(task_masks, self.pre_masks)
            return loss_dict

        else:
            pre_masks = self.load_task_masks(self.save_path)
            features, _, _ = self.backbone(images.tensor, pre_masks)# for testing:  [dict(list(zip(self._out_features, results))), None], 0, None
            # Prediction.
            results = []            
            for i, feature in enumerate(features):
                # Prepare Proposals.
                if self.proposal_generator is not None:
                    proposals, _ = self.proposal_generator(images, feature, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                result = self.roi_heads(images, feature, proposals, None)
                results.append(result)
            # import pdb; pdb.set_trace()
            if do_postprocess:
                processed_results = []
                for result in results:# [N for task [bs * outputs]]
                    for results_per_image, input_per_image, image_size in zip(result, batched_inputs, images.image_sizes):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        r = detector_postprocess(results_per_image[0], height, width)
                        processed_results.append({"instances": r})
                # multi instances to one instance
                processed_results = self.merge_results(processed_results)
                return processed_results
            else:
                return results
    
    def merge_results(self, results):
        saved_boxes = None
        saved_cls = None
        saved_scores = None
        
        for result in results:
            result = result['instances']
            if len(result) <= 0:
                continue
            scores = result.get('scores')
            pred_boxes = result.get('pred_boxes').tensor
            pred_classes = result.get('pred_classes')
            if saved_boxes is None:
                saved_boxes = pred_boxes
                saved_cls = pred_classes
                saved_scores = scores
            else:
                saved_boxes = torch.cat([saved_boxes, pred_boxes],dim=0)
                saved_cls = torch.cat([saved_cls, pred_classes],dim=0)
                saved_scores = torch.cat([saved_scores, scores],dim=0)
        if isinstance(saved_boxes, torch.Tensor):
            pred = Instances(results[0]['instances']._image_size, **{'pred_boxes': Boxes(saved_boxes), 'pred_classes': saved_cls, 'scores':saved_scores}) 
        else:
            pred = Instances(results[0]['instances']._image_size, **{'pred_boxes': Boxes(torch.tensor([]).to(self.device)), 'pred_classes': torch.tensor([]).to(self.device), "scores": torch.tensor([])}) 
        return [{'instances':pred},]

    def merge_result(self, results, gt_results):
        result = results
        tmp = copy.deepcopy(gt_results) # fields = 'gt_boxes' 'gt_classes'
        scores = result.get('scores').cpu()
        pred_boxes = result.get('pred_boxes').tensor.cpu()
        pred_classes = result.get('pred_classes').cpu()
        gt_boxes = tmp.get('gt_boxes').tensor.cpu()
        gt_classes = tmp.get('gt_classes').cpu()
        saved_boxes = None
        saved_cls = None
        # print(pred_boxes)
        for i, score in enumerate(scores):
            if score >= 0.8:
                gt_boxes = torch.cat([gt_boxes, pred_boxes[i].unsqueeze(0)],dim=0)
                gt_classes = torch.cat([gt_classes, pred_classes[i].unsqueeze(0)],dim=0)
                if saved_boxes is None:
                    saved_boxes = pred_boxes[i].unsqueeze(0)
                    saved_cls = pred_classes[i].unsqueeze(0)
                else:
                    saved_boxes = torch.cat([saved_boxes, pred_boxes[i].unsqueeze(0)],dim=0)
                    saved_cls = torch.cat([saved_cls, pred_classes[i].unsqueeze(0)],dim=0)
        
        tmp = Instances(gt_results._image_size, **{'gt_boxes': Boxes(gt_boxes), 'gt_classes': gt_classes})
        if isinstance(saved_boxes, torch.Tensor):
            pred = Instances(gt_results._image_size, **{'gt_boxes': Boxes(saved_boxes), 'gt_classes': saved_cls}) 
        else:
            pred = Instances(gt_results._image_size, **{'gt_boxes': Boxes(torch.tensor([])), 'gt_classes': torch.tensor([])}) 
        return tmp, pred
        

    def old_inference(self, images, pre_masks, batched_inputs):
        features, bottom_up_features, _ = self.dis_backbone(images.tensor, pre_masks)
        results = [] 
        for i, feature in enumerate(features): # i for task_id
            proposals, _ = self.dis_proposal_generator(images, feature, None)
            result = self.dis_roi_heads(images, feature, proposals, None)
            results.append(result)
        # multi instances to one instance
        return features, bottom_up_features, results
        # return features, results

        
    def save_task_masks(self, task_masks, pre_masks=None):
        save_masks = [[x[0].detach().cpu().numpy().tolist(),] for x in task_masks]
        
        if pre_masks is not None:
            tmp = []
            for mak, pre in zip(save_masks, pre_masks):
                tmp.append(mak + [p.detach().cpu().numpy().tolist() for p in pre])
            save_masks = tmp
        save_masks = {'{}'.format(i):v for i, v in enumerate(save_masks) }
        
        
        jsObj = json.dumps(save_masks)  
  
        fileObject = open(self.save_path, 'w')  
        fileObject.write(jsObj)  
        fileObject.close()
        
    def load_task_masks(self, load_path):
        with open(load_path,'r', encoding='UTF-8') as f:
            load_dict = json.load(f)
        task_masks = []
        for k,v in load_dict.items():
            task_masks.append([torch.from_numpy(np.array(m)).cuda().float() for m in v])
        return task_masks




    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        return self.forward(batched_inputs, detected_instances, do_postprocess)

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
# for gen dis
@META_ARCH_REGISTRY.register()
class GeneralizedRCNNDIS(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        task: int=0,
        load_model_path = None,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_backbone= None,
        dis_proposal_generator= None,
        dis_roi_heads= None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # for n,p in self.backbone.named_parameters():
        #     if 'cls_fcs' not in n:
        #         p.requires_grad = False
        # Task
        self.task = task
        self.dis_flag = False
        embed_features = 512
        self.dis_gen = False
        



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        assert cfg.MODEL.TASK > 0
            
        dis_backbone = build_backbone(cfg)
        return {
                "backbone": backbone,
                "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
                "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
                "input_format": cfg.INPUT.FORMAT,
                "vis_period": cfg.VIS_PERIOD,
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "task" : cfg.MODEL.TASK,
                "load_model_path" : cfg.MODEL.WEIGHTS,
                "dis_backbone": dis_backbone,
                "dis_proposal_generator": build_proposal_generator(cfg, dis_backbone.output_shape()),
                "dis_roi_heads": build_roi_heads(cfg, dis_backbone.output_shape()),
            }


    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    def dis_gen_model(self):
        # for p in self.roi_heads.parameters():
        #     p.requires_grad = False
        self.dis_backbone = copy.deepcopy(self.backbone)
        self.dis_proposal_generator = copy.deepcopy(self.proposal_generator)
        self.dis_roi_heads = copy.deepcopy(self.roi_heads)
        for p in self.dis_backbone.parameters():
                p.requires_grad = False
        for p in self.dis_roi_heads.parameters():
            p.requires_grad = False
        for p in self.dis_proposal_generator.parameters():
            p.requires_grad = False
        self.dis_gen = True

        

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], detected_instances = None, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images = self.preprocess_image(batched_inputs)
        # if isinstance(images, (list, torch.Tensor)):
        #     images = nested_tensor_from_tensor_list(images)
        
        
        
        # Feature Extraction.
        if self.training:
            if self.dis_gen is False:
                self.dis_gen_model()
            if self.task > 0 and self.dis_gen:
                self.dis_roi_heads.eval()
                self.dis_backbone.eval()
                self.dis_proposal_generator.eval()
                # self.roi_heads.eval()
                
            dis_loss = []
            old_features, old_backbone_features = self.dis_backbone(images.tensor)
            gt_news = [x['instances'].to(self.device) for x in batched_inputs]
            
            features, backbone_features = self.backbone(images.tensor)# src :[now_feat, [old_mask_feat0, old_mask_feat1..] ]
            
            pre_task_flag = self.task > 0
            if pre_task_flag:
                if len(features) > 0:
                    pre_feats = features[1:]
                    for sr, dis_sr in zip(pre_feats, old_features):
                        for k,v in sr.items():
                            dis_loss.append(F.l1_loss(sr[k], dis_sr[k].detach()))
                for k,vs in backbone_features.items():
                    for i in range(1, len(vs)):
                        dis_loss.append(F.l1_loss(vs[i], old_backbone_features[k][i-1].detach()))
            loss_dict = {}
            for i, feature in enumerate(features): # i for task index
                # Prepare Proposals.
                if self.proposal_generator is not None:
                    proposals, proposal_losses = self.proposal_generator(images, feature, gt_news)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                if i == 0:
                    _, detector_losses = self.roi_heads(images, feature, proposals, gt_news)
                    for k,v in detector_losses.items():
                        if k not in loss_dict.keys():
                            loss_dict[k] = v
                        else:
                            loss_dict[k] += v
                    # print(f'i:{i}', loss_dict)
                else:
                    _, detector_losses = self.roi_heads(images, feature, proposals, gt_news) # -1 for last stage and i for task index
                    for k,v in detector_losses.items():
                        if k not in loss_dict.keys():
                            loss_dict[k] = v
                        else:
                            loss_dict[k] += v
                    # print(detector_losses)
                # print(f'i:{i}', loss_dict)
                
            loss_dict.update(proposal_losses)
            tmp = sum(dis_loss) / len(dis_loss) #* 0.5
            if not torch.isinf(tmp):
                loss_dict['loss_dis'] = tmp
            
            return loss_dict

        else:
            features, _ = self.backbone(images.tensor)# for testing:  [dict(list(zip(self._out_features, results))), None], 0, None
            # Prediction.
            results = []            
            for i, feature in enumerate(features):
                # Prepare Proposals.
                if self.proposal_generator is not None:
                    proposals, _ = self.proposal_generator(images, feature, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                result = self.roi_heads(images, feature, proposals, None)
                results.append(result)
            # import pdb; pdb.set_trace()
            if do_postprocess:
                processed_results = []
                for result in results:# [N for task [bs * outputs]]
                    for results_per_image, input_per_image, image_size in zip(result, batched_inputs, images.image_sizes):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        r = detector_postprocess(results_per_image[0], height, width)
                        processed_results.append({"instances": r})
                # multi instances to one instance
                processed_results = self.merge_results(processed_results)
                return processed_results
            else:
                return results
    
    def merge_results(self, results):
        saved_boxes = None
        saved_cls = None
        saved_scores = None
        
        for result in results:
            result = result['instances']
            if len(result) <= 0:
                continue
            scores = result.get('scores')
            pred_boxes = result.get('pred_boxes').tensor
            pred_classes = result.get('pred_classes')
            if saved_boxes is None:
                saved_boxes = pred_boxes
                saved_cls = pred_classes
                saved_scores = scores
            else:
                saved_boxes = torch.cat([saved_boxes, pred_boxes],dim=0)
                saved_cls = torch.cat([saved_cls, pred_classes],dim=0)
                saved_scores = torch.cat([saved_scores, scores],dim=0)
        if isinstance(saved_boxes, torch.Tensor):
            pred = Instances(results[0]['instances']._image_size, **{'pred_boxes': Boxes(saved_boxes), 'pred_classes': saved_cls, 'scores':saved_scores}) 
        else:
            pred = Instances(results[0]['instances']._image_size, **{'pred_boxes': Boxes(torch.tensor([]).to(self.device)), 'pred_classes': torch.tensor([]).to(self.device), "scores": torch.tensor([])}) 
        return [{'instances':pred},]

    def merge_result(self, results, gt_results):
        result = results
        tmp = copy.deepcopy(gt_results) # fields = 'gt_boxes' 'gt_classes'
        scores = result.get('scores').cpu()
        pred_boxes = result.get('pred_boxes').tensor.cpu()
        pred_classes = result.get('pred_classes').cpu()
        gt_boxes = tmp.get('gt_boxes').tensor.cpu()
        gt_classes = tmp.get('gt_classes').cpu()
        saved_boxes = None
        saved_cls = None
        # print(pred_boxes)
        for i, score in enumerate(scores):
            if score >= 0.8:
                gt_boxes = torch.cat([gt_boxes, pred_boxes[i].unsqueeze(0)],dim=0)
                gt_classes = torch.cat([gt_classes, pred_classes[i].unsqueeze(0)],dim=0)
                if saved_boxes is None:
                    saved_boxes = pred_boxes[i].unsqueeze(0)
                    saved_cls = pred_classes[i].unsqueeze(0)
                else:
                    saved_boxes = torch.cat([saved_boxes, pred_boxes[i].unsqueeze(0)],dim=0)
                    saved_cls = torch.cat([saved_cls, pred_classes[i].unsqueeze(0)],dim=0)
        
        tmp = Instances(gt_results._image_size, **{'gt_boxes': Boxes(gt_boxes), 'gt_classes': gt_classes})
        if isinstance(saved_boxes, torch.Tensor):
            pred = Instances(gt_results._image_size, **{'gt_boxes': Boxes(saved_boxes), 'gt_classes': saved_cls}) 
        else:
            pred = Instances(gt_results._image_size, **{'gt_boxes': Boxes(torch.tensor([])), 'gt_classes': torch.tensor([])}) 
        return tmp, pred
        

    def old_inference(self, images, batched_inputs):
        features, bottom_up_features = self.dis_backbone(images.tensor, batched_inputs)
        # results = [] 
        # for i, feature in enumerate(features): # i for task_id
        #     proposals, _ = self.dis_proposal_generator(images, feature, None)
        #     result = self.dis_roi_heads(images, feature, proposals, None)
        #     results.append(result)
        # multi instances to one instance
        return features, bottom_up_features, _
        # return features, results

        
    




    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        return self.forward(batched_inputs, detected_instances, do_postprocess)

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

