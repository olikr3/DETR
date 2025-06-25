import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class BipartiteBoxMatcher(nn.Module):
    """
    Computes optimal assignment between predicted and ground truth boxes using a bipartite matching algorithm.
    
    The matching considers three factors: class prediction accuracy, box coordinates, and box overlap (GIoU).
    For each batch item, it finds the optimal 1-to-1 matching between predictions and ground truths.
    """
    
    def __init__(self, class_weight: float = 1.0, box_l1_weight: float = 1.0, giou_weight: float = 1.0):
        """
        Initialize the matcher with cost weights.
        
        Args:
            class_weight: Weight for classification error in matching cost
            box_l1_weight: Weight for L1 distance between box coordinates
            giou_weight: Weight for generalized IoU between boxes
        """
        super().__init__()
        self.class_weight = class_weight
        self.box_l1_weight = box_l1_weight
        self.giou_weight = giou_weight
        
        if class_weight == 0 and box_l1_weight == 0 and giou_weight == 0:
            raise ValueError("At least one cost weight must be non-zero")

    @torch.no_grad()
    def forward(self, predictions, targets):
        """
        Perform matching between predictions and ground truth boxes.
        
        Args:
            predictions: Dictionary containing:
                - pred_logits: Classification logits [batch, num_queries, num_classes]
                - pred_boxes: Predicted boxes [batch, num_queries, 4]
                
            targets: List of target dictionaries (one per batch item) containing:
                - labels: Class labels [num_targets]
                - boxes: Target box coordinates [num_targets, 4]
                
        Returns:
            List of tuples (pred_indices, target_indices) for each batch item,
            representing the optimal matches.
        """
        batch_size, num_predictions = predictions["pred_logits"].shape[:2]
        
        # Flatten predictions for batch processing
        class_probs = predictions["pred_logits"].flatten(0, 1).softmax(-1)
        pred_boxes = predictions["pred_boxes"].flatten(0, 1)
        
        # Concatenate all targets
        target_labels = torch.cat([t["labels"] for t in targets])
        target_boxes = torch.cat([t["boxes"] for t in targets])
        
        # Calculate matching cost components
        classification_cost = -class_probs[:, target_labels]  # Negative log-like
        box_coord_cost = torch.cdist(pred_boxes, target_boxes, p=1)  # L1 distance
        
        # Convert boxes to xyxy format for GIoU
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        box_overlap_cost = -generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        
        # Combine costs with weights
        total_cost = (
            self.box_l1_weight * box_coord_cost +
            self.class_weight * classification_cost +
            self.giou_weight * box_overlap_cost
        )
        
        # Reshape cost matrix and split by batch items
        cost_matrix = total_cost.view(batch_size, num_predictions, -1).cpu()
        target_sizes = [len(t["boxes"]) for t in targets]
        
        # Solve assignment problem for each batch item
        matches = []
        for i, (cost, size) in enumerate(zip(cost_matrix.split(target_sizes, -1), target_sizes)):
            pred_idx, target_idx = linear_sum_assignment(cost[i])
            matches.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(target_idx, dtype=torch.int64)
            ))
            
        return matches


def create_matcher(config):
    """Factory function to create a matcher instance from configuration."""
    return BipartiteBoxMatcher(
        class_weight=config.set_cost_class,
        box_l1_weight=config.set_cost_bbox,
        giou_weight=config.set_cost_giou
    )
