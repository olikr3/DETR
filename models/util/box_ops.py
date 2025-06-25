import torch
import torchvision.ops.boxes as box_ops

def box_cxcywh_to_xyxy(x):
    """Convert bounding box format from [center_x, center_y, width, height] to [x1, y1, x2, y2]"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """Convert bounding box format from [x1, y1, x2, y2] to [center_x, center_y, width, height]"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x1, y1, x2, y2] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # Compute standard IoU
    iou = box_ops.box_iou(boxes1, boxes2)
    
    # Compute the smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Compute GIoU
    union = (box_ops.box_area(boxes1)[:, None] + 
             box_ops.box_area(boxes2) - iou * box_ops.box_area(boxes1)[:, None])
    giou = iou - (area - union) / area
    
    return giou