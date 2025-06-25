import torch
import torch.utils.data
import torchvision
from pycocotools.coco import COCO
from PIL import Image
import os

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Custom COCO dataset that works with the subset you created.
    Inherits from torchvision.datasets.CocoDetection but adds proper image loading.
    """
    
    def __init__(self, img_folder, ann_file, transforms=None):
        """
        Args:
            img_folder (string): Path to the folder with images
            ann_file (string): Path to the COCO annotation file
            transforms (callable, optional): Optional transforms to be applied
        """
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, target) where target is a dictionary containing:
                - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format
                - labels (Int64Tensor[N]): the class label for each ground-truth box
                - image_id (Int64Tensor[1]): the image ID
                - area (Tensor[N]): the area of each box
                - iscrowd (UInt8Tensor[N]): iscrowd flag
        """
        coco = self.coco
        img_id = self.ids[idx]
        
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        
        # Load image
        img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        
        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # Convert annotations to target format
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # Skip invalid annotations
            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                continue
                
            # Convert COCO bbox format (x,y,w,h) to (x1,y1,x2,y2)
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.as_tensor([img_id])
        target['area'] = torch.as_tensor(areas, dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.uint8)
        
        # Apply transforms if any
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

def collate_fn(batch):
    """
    Custom collate function to handle variable numbers of objects per image
    """
    return tuple(zip(*batch))