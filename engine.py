import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm, matcher):
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = compute_loss(outputs, targets, matcher)
        losses = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        total_loss += losses.item()
    
    return {"train_loss": total_loss / len(data_loader)}

def evaluate(model, data_loader, device, matcher):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            loss_dict = compute_loss(outputs, targets, matcher)
            losses = sum(loss_dict.values())
            total_loss += losses.item()
    
    return {"val_loss": total_loss / len(data_loader)}

def compute_loss(outputs, targets, matcher):
    # Get matching between predictions and ground truth
    matches = matcher(outputs, targets)
    
    # Compute classification loss (focal loss)
    src_logits = outputs['pred_logits']
    target_classes = torch.full(src_logits.shape[:2], 0, 
                              dtype=torch.int64, device=src_logits.device)
    
    for i, (pred_idx, target_idx) in enumerate(matches):
        target_classes[i, pred_idx] = targets[i]['labels'][target_idx]
    
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
    
    # Compute bounding box loss (L1 + GIoU)
    src_boxes = outputs['pred_boxes']
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, matches)], dim=0)
    
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        box_cxcywh_to_xyxy(src_boxes),
        box_cxcywh_to_xyxy(target_boxes)
    ))
    
    losses = {
        'loss_ce': loss_ce,
        'loss_bbox': loss_bbox.sum() / len(targets),
        'loss_giou': loss_giou.sum() / len(targets)
    }
    
    # Auxiliary losses if any
    if 'aux_outputs' in outputs:
        for i, aux_outputs in enumerate(outputs['aux_outputs']):
            aux_losses = compute_loss(aux_outputs, targets, matcher)
            losses.update({f'{k}_{i}': v for k, v in aux_losses.items()})
    
    return losses