import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models.embedding import build_transformer
from models.matcher import create_matcher
from models.backbone import build_backbone
from datasets.coco import CocoDetection
from models.detr import DETR
from engine import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('DETR training and evaluation script', add_help=False)
    
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    
    # Model parameters
    parser.add_argument('--num_classes', default=5, type=int)  # Matches your 5 categories
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    
    # Matcher parameters
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', default='./data/coco', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    
    return parser

def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=True
    )
    
    matcher = create_matcher(args)
    
    return model, matcher

def main(args):
    device = torch.device(args.device)
    
    # Data transforms
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CocoDetection(
        img_folder=f'{args.dataset_dir}/train2017_subset',
        ann_file=f'{args.dataset_dir}/annotations_subset/instances_train2017.json',
        transforms=normalize
    )
    
    val_dataset = CocoDetection(
        img_folder=f'{args.dataset_dir}/val2017_subset',
        ann_file=f'{args.dataset_dir}/annotations_subset/instances_val2017.json',
        transforms=normalize
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Build model
    model, matcher = build_model(args)
    model.to(device)
    
    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            args.clip_max_norm,
            matcher
        )
        
        lr_scheduler.step()
        
        # Evaluate on validation set
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            eval_stats = evaluate(model, val_loader, device, matcher)
            print(f"Validation results at epoch {epoch}: {eval_stats}")
    
    print("Training complete!")

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])  # Images
    return tuple(batch)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)