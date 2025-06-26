# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample
)

from train.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list



def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch

    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.sum(1) / max(1, loss.shape[1])

    return loss.sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_with_weight_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch

    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    weights = weights.flatten(1)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    return loss.sum() / max(num_masks, 0.5)


sigmoid_ce_with_weight_loss_jit = torch.jit.script(
    sigmoid_ce_with_weight_loss
)  # type: torch.jit.ScriptModule


def dice_coefficient_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. Nx...
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs**2).sum(-1) + (targets**2).sum(-1)
    loss = 1 - numerator / denominator.clamp(min=1e-6)
    return loss.sum() / num_masks


dice_coefficient_loss_jit = torch.jit.script(
    dice_coefficient_loss
)  # type: torch.jit.ScriptModule


def focal_conf_sigmoid_loss(inputs, targets, alpha=0.5, gamma=2, is_cls=False):
    """
    Focal loss but using sigmoid like the original paper, where alpha balances the positive or negative samples,
    and gamma controls the easy or difficult samples.
    inputs: ... x N x K, masked these classes do not belong to this dataset
    targets: ... x N x k, one-hot embedding of targets
    Note: To make learnables mesh easier, the network predicts K+1 class confidences in this mode,
          but it has been masked with '-inf'.
    """
    # inputs = inputs.flatten(0, -2)    # N, num_classes
    # targets = targets.flatten(0, -2)  # N, num_classes

    targets_pm = targets * 2 - 1  # -1 if non-target classes, +1 if target class

    logpt = F.logsigmoid(inputs * targets_pm)  # note: 1 - sigmoid(x) = sigmoid(-x)
    pt = logpt.exp()

    # multi-class focal losses
    num_neg_classes = torch.logical_not(targets).sum(-1).clamp(min=1).unsqueeze(-1)
    at = alpha * targets + (1 - alpha) * (1 - targets)

    loss = -at * (1 - pt) ** gamma * logpt
    loss = loss.sum(-1)
    if not is_cls:
        loss = loss / targets.sum(-1).clamp(min=1)

    return loss

def contrastive_loss(inputs, targets, topk=20):
    if inputs.nelement() == 0:
        return inputs[:0].sum().detach()

    inputs = inputs.flatten(1)    # N, K
    targets = targets.flatten(1)  # N, K
    keep = targets.sum(1) > 0
    inputs = inputs[keep]
    targets = targets[keep]
    N = inputs.shape[0]
    topk = min(min(topk, 20), 3*N)

    pos_indices = targets.argmax(1)
    pos_inputs = inputs[torch.arange(N), pos_indices]

    pos_inputs_mean = (inputs * targets).sum(-1) / targets.sum(-1).clamp(min=1)
    pos_inputs = torch.stack([pos_inputs, pos_inputs_mean], dim=1)  # N K_pos
    
    neg_indices = torch.nonzero(targets.sum(0) > 0.).reshape(-1)
    bg_indices = torch.nonzero(targets.sum(0) == 0.).reshape(-1)
    neg_indices = neg_indices[torch.randperm(len(neg_indices))[:int(0.75*topk)]]
    bg_indices = bg_indices[torch.randperm(len(bg_indices))[:int(0.25*topk)]]
    neg_indices = torch.cat([neg_indices, bg_indices]).sort()[0]
    
    inputs = inputs[:, neg_indices]  # N K_neg
    targets = targets[:, neg_indices]  # N K_neg

    negpos_inputs = (inputs[:, :, None] - pos_inputs[:, None]) * torch.logical_not(targets)[:, :, None]  # N K_neg K_pos 
    negpos_inputs = negpos_inputs.clamp(max=10.).exp() * torch.logical_not(targets)[:, :, None]  # N K_neg K_pos
    
    # loss = torch.logsumexp(inputs_negpos, dim=-1)
    loss = (1 + torch.sum(negpos_inputs.flatten(1), dim=-1)).log()
    loss = loss.sum() / max(N, 1)

    return loss

def contrastive_aux_loss(inputs, targets, topk=20):
    if inputs.nelement() == 0:
        return inputs[:0].sum().detach()
    
    # assert inputs.min() >= -1 and inputs.max() <= 1, f'invalid values: min {inputs.min()} and max {inputs.max()}'
    inputs = inputs.flatten(1)    # N, K
    targets = targets.flatten(1)  # N, K
    N = inputs.shape[0]
    topk = min(min(topk, 20), 3*N)

    pos_indices = torch.nonzero(targets.sum(0) > 0.).reshape(-1)
    pos_indices = pos_indices[torch.randperm(len(pos_indices))[:int(0.75*topk)]]
    bg_indices = torch.nonzero(targets.sum(0) == 0.).reshape(-1)
    bg_indices = bg_indices[torch.randperm(len(bg_indices))[:int(0.25*topk)]]
    indices = torch.cat([pos_indices, bg_indices]).sort()[0]

    inputs = inputs[:, indices].clamp(min=0.)  # N K_neg
    targets = targets[:, indices]  # N K_neg
    loss = F.smooth_l1_loss(inputs, targets, reduction='sum')
    loss = loss / max(N, 1)

    return loss 





def smooth_l1_loss(input, target, reduction='sum'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond**2 / pos_inds.shape[0]
    else:
        loss = input * 0.0

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise NotImplementedError



def calculate_iou(src_masks_logits, target_masks, threshold=0.5):
    """
    Calculate the Intersection over Union (IoU) between predicted masks and target masks.
    
    Parameters:
    src_masks_logits (torch.Tensor): Predicted mask logits with shape [num_queries, batch_size, height, width].
    target_masks (torch.Tensor): Ground truth masks with shape [num_queries, batch_size, height, width].
    threshold (float): Threshold to convert predicted masks to binary masks.
    
    Returns:
    torch.Tensor: The IoU for each query and batch with shape [num_queries, batch_size].
    """
    # Apply sigmoid to logits to get predicted masks
    pred_masks = torch.sigmoid(src_masks_logits)
    
    # Downsample target masks to match the shape of predicted masks using nearest neighbor interpolation for speed
    target_masks = F.interpolate(target_masks, size=(src_masks_logits.shape[2], src_masks_logits.shape[3]), mode='nearest')
    
    # Convert predicted masks to binary masks using the threshold
    pred_masks_binary = (pred_masks > threshold).float()
    target_masks_binary = (target_masks > threshold).float()
    
    # Calculate intersection and union using logical operations
    intersection = torch.logical_and(pred_masks_binary, target_masks_binary).sum(dim=(2, 3))
    union = torch.logical_or(pred_masks_binary, target_masks_binary).sum(dim=(2, 3))
    
    # Compute IoU
    iou = intersection / union
    iou = iou.detach()
    
    # Return IoU for each query and batch
    return iou




def get_mask_centers(high_res_multimasks):
    # Threshold the masks to get boolean masks
    bool_masks = high_res_multimasks.bool() # Shape: [N, H, W]
    N, H, W = bool_masks.shape
    if N==0:
        import pdb;pdb.set_trace()

    # Create coordinate grids
    device = bool_masks.device
    y_coords = torch.arange(H, device=device).unsqueeze(1).expand(H, W)  # Shape: [H, W]
    x_coords = torch.arange(W, device=device).unsqueeze(0).expand(H, W)  # Shape: [H, W]

    # Expand coordinates to match the batch size
    x_coords = x_coords.unsqueeze(0)  # Shape: [1, H, W]
    y_coords = y_coords.unsqueeze(0)  # Shape: [1, H, W]

    # Compute sums over the masks
    sum_x = (bool_masks * x_coords).view(N, -1).sum(dim=1)  # Shape: [N]
    sum_y = (bool_masks * y_coords).view(N, -1).sum(dim=1)  # Shape: [N]
    counts = bool_masks.view(N, -1).sum(dim=1)              # Shape: [N]

    # Avoid division by zero by replacing zeros with ones temporarily
    counts_safe = counts.clone()
    counts_safe[counts_safe == 0] = 1  # Replace zero counts with one to prevent division by zero

    # Compute centers
    centers_x = sum_x / counts_safe  # Shape: [N]
    centers_y = sum_y / counts_safe  # Shape: [N]

    # Set centers to (0, 0) where counts are zero
    centers_x[counts == 0] = 0
    centers_y[counts == 0] = 0

    # Stack the centers into a tensor of shape [N, 2]
    centers = torch.stack([centers_x, centers_y], dim=1)

    return centers

def calculate_uncertainty(logits):
    """
    We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))



from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    Modified from torchvision to also return the union
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)

class BoxVISTeacherSetPseudoMask(nn.Module):
    def __init__(self, matcher):
        """Create the criterion.
        Parameters:
            matcher: matching objects between targets and proposals
        """
        super().__init__()
        self.matcher = matcher

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc.
                      The mask in targets is generated mask via bounding boxes
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        targets = self.set_pseudo_mask(indices, outputs_without_aux, targets)
        for bs, target in enumerate(targets):
            tgt_mask_box = target["masks"]
            tgt_mask_pseudo = target["masks_pseudo"]
            tgt_mask_pseudo_score = target["mask_pseudo_scores"]

            tgt_h, tgt_w = tgt_mask_box.shape[-2:]
            tgt_mask_pseudo = F.interpolate(tgt_mask_pseudo, (tgt_h, tgt_w),
                                            mode='bilinear', align_corners=False)  # cQ, T, Ht, Wt

            #  project term --------------------------
            tgt_mask_pseudo_y = tgt_mask_pseudo.sigmoid().max(dim=-2, keepdim=True)[0].flatten(1)
            tgt_mask_box_y = tgt_mask_box.max(dim=-2, keepdim=True)[0].flatten(1)
            numerator = 2 * (tgt_mask_pseudo_y * tgt_mask_box_y).sum(-1)
            denominator = (tgt_mask_pseudo_y ** 2).sum(-1) + (tgt_mask_box_y ** 2).sum(-1)
            mask_proj_y = numerator / denominator.clamp(min=1e-6)

            tgt_mask_pseudo_x = tgt_mask_pseudo.sigmoid().max(dim=-1, keepdim=True)[0].flatten(1)
            tgt_mask_box_x = tgt_mask_box.max(dim=-1, keepdim=True)[0].flatten(1)
            numerator = 2 * (tgt_mask_pseudo_x * tgt_mask_box_x).sum(-1)
            denominator = (tgt_mask_pseudo_x ** 2).sum(-1) + (tgt_mask_box_x ** 2).sum(-1)
            mask_proj_x = numerator / denominator.clamp(min=1e-6)

            mask_proj_score = 0.5 * (mask_proj_x + mask_proj_y)

            target["mask_pseudo_scores"] = tgt_mask_pseudo_score * mask_proj_score
            target["masks_pseudo"] = tgt_mask_box * tgt_mask_pseudo.sigmoid()

        return targets

    def set_pseudo_mask(self, indices, outputs, targets):
        src_masks = outputs["pred_masks"].clone().detach()  # B, cQ, T, Hp, Wp
        src_logits = outputs["pred_logits"].softmax(dim=-1).clone().detach()  # B, cQ, k

        for i, ((src_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
            assert len(tgt_idx) == target["masks"].shape[0]
            tgt_idx, tgt_idx_sorted = tgt_idx.sort()
            src_idx = src_idx[tgt_idx_sorted]
            tgt_labels = target["labels"]
            target["mask_pseudo_scores"] = src_logits[i, src_idx, tgt_labels]
            target["masks_pseudo"] = src_masks[i, src_idx]

        return targets


class VideoSetCriterionClsagIoU(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_frames, 
                 num_points, oversample_ratio, importance_sample_ratio, 
                 use_ctt_loss=True, max_num_masks: int=50, boxvis_enabled=False, 
                 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses
            boxvis_enabled: It controls the annotation types: pixel-wise or box-level annotations for VIS task
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_frames = num_frames
        self.use_ctt_loss = use_ctt_loss

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.max_num_masks = max_num_masks

        # box-supervised video instance segmentation
        self.boxvis_enabled = boxvis_enabled
    
    def loss_labels_clip(self, outputs, targets, indices, num_masks, l_layer):
        loss = []
        num_objs = []

        out_logits = outputs['pred_logits']
        for t, logits, (src_i, tgt_i) in zip(targets, out_logits, indices):
            if t['task'] != 'grounding':
                # if t['dataset_name'] not in combined_datasets_category_info:
                #     continue 
                # num_classes, start_idx = combined_datasets_category_info[t['dataset_name']]
                # logits = logits[:, start_idx:start_idx + num_classes]

                tgt_classes = torch.full(
                    logits.shape, 0, dtype=torch.int64, device=logits.device
                )
                # tgt_labels = t["labels"][tgt_i] - 1  # starts from 1
                tgt_labels = t["labels"][tgt_i] * 0 + 1  # Foreground

                tgt_classes[src_i, tgt_labels] = 1
                # tgt_classes[:,0] = 1 - tgt_classes[:,1] # NOTE binary classification

                loss_focal = focal_conf_sigmoid_loss(logits, tgt_classes, is_cls=True)
                loss_focal = loss_focal.sum() / max(len(tgt_labels), 1)

                # cross-entroy loss
                if len(tgt_labels):
                    loss_ce = F.cross_entropy(logits[src_i], tgt_labels)
                    loss.append(loss_focal + loss_ce)
                else:
                    loss.append(loss_focal)
                num_objs.append(len(tgt_labels))

        if len(loss) == 0:
            return {"loss_ce": out_logits[:0].sum().detach()}
        else:
            weighted_loss = [num_obj / max(sum(num_objs), 1) * loss_ for num_obj, loss_ in zip(num_objs, loss)]
            return {"loss_ce": sum(weighted_loss)}
            # return {"loss_ce": sum(loss) / len(loss)}
    
    def loss_reid(self, outputs, targets, indices, num_masks, l_layer):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes],
        we do not take the background class into account
        outputs["pred_embds"]: B x Q x T x C
        """
        assert "pred_embds" in outputs 

        if outputs["pred_embds"].nelement() == 0:
            if "pred_embds_prompt" in outputs:
                return {
                    "loss_reid": outputs["pred_embds"][:0].sum().detach(),
                    "loss_reid_aux": outputs["pred_embds"][:0].sum().detach(),
                    "loss_reid_l2p": outputs["pred_embds"][:0].sum().detach(),
                    "loss_reid_l2p_aux": outputs["pred_embds"][:0].sum().detach()
                }
            else:
                return {
                    "loss_reid": outputs["pred_embds"][:0].sum().detach(),
                    "loss_reid_aux": outputs["pred_embds"][:0].sum().detach()
                }
    
        bs = len(targets)
        device = outputs["pred_embds"].device

        src_indices = self._get_src_permutation_idx(indices)
        pred_embds = outputs["pred_embds"]                  # B Q T C
        src_embds = pred_embds[src_indices].flatten(0, -2)  # N_srcT x C
        tgt_ids = torch.cat([t['ids'][tgt_i] for t, (_, tgt_i) in zip(targets, indices)]).to(device).flatten()  # N_src T 
        vid_ids = torch.cat([torch.ones_like(tgt_i)[:, None].repeat(1, self.num_frames) * i 
                             for i, (_, tgt_i) in enumerate(indices)]).to(device).flatten()                     # N_src T 

        keep = tgt_ids >= 0
        src_embds = src_embds[keep]
        tgt_ids = tgt_ids[keep]
        vid_ids = vid_ids[keep]

        src_sim = torch.mm(src_embds, src_embds.t()) / math.sqrt(src_embds.shape[-1])  # N_srcT x N_srcT
        tgt_classes = (tgt_ids[:, None] == tgt_ids[None]) & (vid_ids[:, None] == vid_ids[None])
        tgt_classes = tgt_classes.float()
        
        if not self.use_ctt_loss:
            loss_focal = focal_conf_sigmoid_loss(src_sim, tgt_classes, is_cls=False)
            loss_focal = loss_focal / max(self.num_frames * bs, 1)
            loss_focal = loss_focal.sum() / max(num_masks * bs, 1)
        else:
            loss_focal = contrastive_loss(src_sim, tgt_classes)
        
        # aux loss
        sim_aux = torch.einsum(
            'qc,kc->qk', F.normalize(src_embds, p=2, dim=-1),
            F.normalize(src_embds, p=2, dim=-1)
        )
        sim_aux = sim_aux[tgt_classes.sum(-1) > 0]
        tgt_classes = tgt_classes[tgt_classes.sum(-1) > 0]
        loss_aux = contrastive_aux_loss(sim_aux, tgt_classes) 

        loss =  {"loss_reid": loss_focal, "loss_reid_aux": loss_aux}

        if "pred_embds_prompt" in outputs:
            loss_l2p = self.loss_reid_l2p(outputs, targets, indices, num_masks, l_layer)
            loss.update(loss_l2p)
        
        # store query embds to calculate inter-clip reid loss for stage3
        if len(targets) == 1:
            assert len(targets) == 1, 'Only support bacth size = 1'
            targets[0]['src_embds'][l_layer].append(src_embds.clone())
            targets[0]['tgt_ids'][l_layer].append(tgt_ids)

        return loss
    
    def loss_reid_l2p(self, outputs, targets, indices, num_masks, l_layer):
        """Reid loss from learnable to prompt queries (NLL)
        for detection task, the reid loss is based on whether they have same classes;
        for grounding task, the reid loss is base on whether the expressions describe the same object or not.
        for sot task, the reid loss is base on whether the visual prompt comes from the same entity
        targets dicts must contain the key "pred_embds_prompt" containing a tensor of dim [nb_target_boxes],
        we do not take the background class into account
        outputs["pred_embds"]: B x Q x T x C
        """
        assert "pred_embds" in outputs and "pred_embds_prompt" in outputs 

        if outputs["pred_embds_prompt"].nelement() == 0:
            return {
                "loss_reid_l2p": outputs["pred_embds_prompt"][:0].sum().detach(),
                "loss_reid_l2p_aux": outputs["pred_embds_prompt"][:0].sum().detach(),
            }
        
        bs = len(targets)
        device = outputs["pred_embds_prompt"].device

        src_indices = self._get_src_permutation_idx(indices)
        pred_embds = outputs["pred_embds"]  # B Q T C
        src_embds = pred_embds[src_indices].flatten(0, -2)    # N_src T x C
        pred_embds_prompt = outputs["pred_embds_prompt"].flatten(0, -2)  # B Q_p T C -> N_src_p T x C
        
        # N_l: the number of the matched masks in all predicted masks BQ_lT
        vid_ids_l = torch.cat([torch.ones_like(tgt_i) * i for i, (_, tgt_i) in enumerate(indices)]).to(device)    
        vid_ids_l = vid_ids_l.unsqueeze(-1).repeat(1, self.num_frames).flatten() 
        num_queries_p = int(pred_embds_prompt.shape[0] / bs)
        vid_ids_p = torch.stack([torch.ones(num_queries_p, device=device) * i for i in range(bs)]).flatten()

        task = targets[0]["task"]
        if task == "detection" and targets[0]["prompt_type"] == "text":
            tgt_ids_l = torch.cat([t['labels'][tgt_i] for t, (_, tgt_i) in zip(targets, indices)]).to(device)  # N_l
            tgt_ids_l = tgt_ids_l.unsqueeze(-1).repeat(1,self.num_frames).flatten()     # N_lT
            tgt_ids_p = torch.cat([t['prompt_gt_labels'] for t in targets]).to(device)  # BQ_p
            tgt_ids_p = tgt_ids_p.unsqueeze(-1).repeat(1,self.num_frames).flatten()     # BQ_pT
            keep_l = tgt_ids_l >= 1
            keep_p = torch.cat([t['prompt_obj_ids'] for t in targets]) >= 0
            keep_p = keep_p.unsqueeze(-1).repeat(1,self.num_frames).flatten().to(device)  # BQ_pT

        else:
            tgt_ids_l = torch.cat([t['ids'][tgt_i] for t, (_, tgt_i) in zip(targets, indices)]).to(device)  # N_lxT
            tgt_ids_l = tgt_ids_l.flatten()   # N_lT
           
            tgt_ids_p = []
            for t in targets:
                valid = t["prompt_obj_ids"] >= 0  # N_p
                tgt_ids_p_cur = t["ids"][t["prompt_obj_ids"]].long()
                tgt_ids_p_cur[~valid] = -1
                tgt_ids_p.append(tgt_ids_p_cur)
            tgt_ids_p = torch.stack(tgt_ids_p).to(device).flatten() # BQ_pT

            keep_l = tgt_ids_l >= 0
            keep_p = tgt_ids_p >= 0
    
        src_embds = src_embds[keep_l]
        tgt_ids_l = tgt_ids_l[keep_l]
        vid_ids_l = vid_ids_l[keep_l]

        pred_embds_prompt = pred_embds_prompt[keep_p]
        tgt_ids_p = tgt_ids_p[keep_p]
        vid_ids_p = vid_ids_p[keep_p]

        tgt_classes = (tgt_ids_l[:, None] == tgt_ids_p[None]) & (vid_ids_l[:, None] == vid_ids_p[None])
        tgt_classes = tgt_classes.float()

        src_sim = torch.mm(src_embds, pred_embds_prompt.t()) / math.sqrt(src_embds.shape[-1])
        loss_focal = contrastive_loss(src_sim, tgt_classes)
        
        # aux loss
        if task == "detection" and targets[0]["prompt_type"] == "text":
            loss_aux = src_embds[:0].sum().detach()
        else:
            sim_aux = torch.einsum(
                'qc,kc->qk', F.normalize(src_embds, p=2, dim=-1), 
                F.normalize(pred_embds_prompt, p=2, dim=-1)
            )
            sim_aux = sim_aux[tgt_classes.sum(-1) > 0]
            tgt_classes = tgt_classes[tgt_classes.sum(-1) > 0]
            loss_aux = contrastive_aux_loss(sim_aux, tgt_classes) 

        # store query embds to calculate inter-clip reid loss for stage3
        if len(targets) == 1 and not (task == "detection" and targets[0]["prompt_type"] == "text"):
            assert len(targets) == 1, 'Only support bacth size = 1'
            targets[0]['src_embds'][l_layer].append(pred_embds_prompt.clone())
            targets[0]['tgt_ids'][l_layer].append(tgt_ids_p)

        return {"loss_reid_l2p": loss_focal, "loss_reid_l2p_aux": loss_aux}
        
    def loss_masks(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, T, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords: NT x 12544 x 2
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_with_box_supervised(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the masks with only box annotations: the projection loss.
        If enabling Teacher Net with EMA, the pseudo mask supervision includes the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, T, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]  # NxTxHpxWp
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)  # NxTxHxW
        tgt_h, tgt_w = target_masks.shape[-2:]

        # the predicted masks are inaccurate in the initial iterations
        h_, w_ = int(tgt_h/2), int(tgt_w/2)
        with torch.no_grad():
            target_masks = F.interpolate(target_masks, (h_, w_),
                                            mode='nearest', align_corners=False)
            src_masks = F.interpolate(src_masks, (h_, w_),
                                        mode='bilinear', align_corners=False)

        # ----------------------- points out of bounding box / project term --------------------------
        # max simply selects the greatest value to back-prop, so max is the identity operation for that one element
        src_masks = src_masks.sigmoid()
        mask_losses_y = dice_coefficient_loss_jit(
            src_masks.max(dim=-2, keepdim=True)[0].flatten(1),
            target_masks.max(dim=-2, keepdim=True)[0].flatten(1),
            num_masks
        )
        mask_losses_x = dice_coefficient_loss_jit(
            src_masks.max(dim=-1, keepdim=True)[0].flatten(1),
            target_masks.max(dim=-1, keepdim=True)[0].flatten(1),
            num_masks
        )
        loss_proj = mask_losses_x + mask_losses_y
        losses = {'loss_mask_proj': loss_proj}

        return losses
    
    def loss_ious(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_logits_iou' or 'pred_iou' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_masks_logits = outputs['pred_masks'][idx]
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        target_iou = calculate_iou(src_masks_logits,target_masks)
        if 'pred_iou' in outputs:
            src_iou = outputs['pred_iou'][idx].squeeze()
        else:
            src_iou = outputs['pred_logits_iou'][idx].sigmoid().squeeze()
        
        target_iou = target_iou.flatten()
        src_iou = src_iou.flatten()
        
        loss_iou = smooth_l1_loss(src_iou, target_iou)

        losses = {}
        losses['loss_iou'] = loss_iou

        return losses

    def loss_points(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_logits_iou' or 'pred_iou' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_masks_logits = outputs['pred_masks'][idx]
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_points = outputs['sam_point_coords'][idx].flatten(0,1)
        if(len(target_masks) == 0):
            return {"loss_point": src_points[:0].sum().detach()}

        target_points = get_mask_centers(target_masks.flatten(0,1))

        target_points = target_points.flatten()
        src_points = src_points.flatten()
        
        loss_point = smooth_l1_loss(src_points, target_points, reduction='mean')

        losses = {}
        losses['loss_point'] = loss_point / 1024

        return losses


    def loss_boxes(self, outputs, targets, indices, num_masks, l_layer):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_logits_iou' or 'pred_iou' in outputs
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes.flatten(0,1)

        src_points = outputs['sam_point_coords'][idx].flatten(0,1)
        if(len(target_boxes) == 0):
            return {"loss_point": src_points[:0].sum().detach()}
        
        valid_mask = target_boxes.sum(dim=-1) != 0  # [N]
        loss_bbox = F.l1_loss(src_points[valid_mask], target_boxes[valid_mask], reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_points[valid_mask]), 
            target_boxes[valid_mask]
        ))
        losses = {}
        losses['loss_point'] = (loss_bbox.sum() + loss_giou.sum()) / num_masks

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, l_layer=9):
        if self.boxvis_enabled:
            loss_map = {
                'labels': self.loss_labels_clip,
                'masks': self.loss_masks_with_box_supervised,
            }
        else:
            loss_map = {
                'labels': self.loss_labels_clip,
                'masks': self.loss_masks,
                'reid': self.loss_reid,
                'ious': self.loss_ious,
                'point': self.loss_boxes,
            }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, l_layer)

    def forward(self, outputs, targets, input_type='video'):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depend on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        if input_type == 'video':
            self.num_frames = outputs['pred_masks'].shape[2]
            num_masks = num_masks * self.num_frames
        elif input_type == 'image':
            num_masks = num_masks * 1
        else:
            raise NotImplementedError

        outputs_without_aux = outputs
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, l_layer=9))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_masks, l_layer=i)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
