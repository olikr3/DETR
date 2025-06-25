#!/bin/bash

# COCO 2017 Subset for Specific Categories (dog, person, car, bottle, chair)

# Exit on error
set -e

# Configuration
DATASET_DIR="./data/coco"
CATEGORIES=("dog" "person" "car" "bottle" "chair")  # Categories to keep
MIN_IMAGES_PER_CAT=1000  # Minimum images per category (will take less if not available)

# Create directory structure
mkdir -p $DATASET_DIR
cd $DATASET_DIR

# Download datasets
echo "Downloading COCO 2017 datasets..."
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip datasets
echo "Unzipping datasets..."
unzip -q train2017.zip -d .
unzip -q val2017.zip -d .
unzip -q annotations_trainval2017.zip -d .

# Create filtered subset
echo "Creating filtered subset for categories: ${CATEGORIES[@]}..."
python3 - <<END
import json
import os
import shutil

# Category names to keep
selected_categories = set(("dog", "person", "car", "bottle", "chair"))

# Load annotations
with open('annotations/instances_train2017.json') as f:
    train_ann = json.load(f)
with open('annotations/instances_val2017.json') as f:
    val_ann = json.load(f)

# Get category IDs for our selected categories
category_map = {}
selected_category_ids = set()
for cat in train_ann['categories']:
    if cat['name'] in selected_categories:
        category_map[cat['id']] = cat['name']
        selected_category_ids.add(cat['id'])

print(f"Selected category IDs: {selected_category_ids}")

def filter_dataset(ann_data, img_dir, output_dir):
    # Find all image IDs with our selected categories
    valid_image_ids = set()
    category_counts = {name: 0 for name in selected_categories}
    
    for ann in ann_data['annotations']:
        if ann['category_id'] in selected_category_ids:
            valid_image_ids.add(ann['image_id'])
            cat_name = category_map[ann['category_id']]
            category_counts[cat_name] += 1
    
    print(f"Category counts before filtering: {category_counts}")
    
    # Get image metadata for valid images
    valid_images = [img for img in ann_data['images'] if img['id'] in valid_image_ids]
    
    # Limit to MIN_IMAGES_PER_CAT per category (while maintaining balance)
    category_image_map = {name: [] for name in selected_categories}
    for img in valid_images:
        img_id = img['id']
        img_anns = [a for a in ann_data['annotations'] if a['image_id'] == img_id]
        for ann in img_anns:
            if ann['category_id'] in selected_category_ids:
                cat_name = category_map[ann['category_id']]
                category_image_map[cat_name].append(img)
                break  # Only count once per image
    
    # Find minimum available across categories
    min_images = min(len(images) for images in category_image_map.values())
    min_images = min(min_images, $MIN_IMAGES_PER_CAT)
    print(f"Taking {min_images} images per category")
    
    # Select balanced subset
    selected_images = set()
    for cat, images in category_image_map.items():
        for img in images[:min_images]:
            selected_images.add(img['id'])
    
    # Filter annotations
    ann_data['images'] = [img for img in ann_data['images'] if img['id'] in selected_images]
    ann_data['annotations'] = [ann for ann in ann_data['annotations'] 
                             if ann['image_id'] in selected_images and 
                             ann['category_id'] in selected_category_ids]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy selected images
    for img in ann_data['images']:
        src = os.path.join(img_dir, img['file_name'])
        dst = os.path.join(output_dir, img['file_name'])
        shutil.copy(src, dst)
    
    return ann_data

# Process both train and val
print("\nProcessing training set...")
train_subset = filter_dataset(train_ann, 'train2017', 'train2017_subset')

print("\nProcessing validation set...")
val_subset = filter_dataset(val_ann, 'val2017', 'val2017_subset')

# Save filtered annotations
os.makedirs('annotations_subset', exist_ok=True)
with open('annotations_subset/instances_train2017.json', 'w') as f:
    json.dump(train_subset, f)
with open('annotations_subset/instances_val2017.json', 'w') as f:
    json.dump(val_subset, f)

print("\nFiltered dataset created successfully!")
END

# Clean up
echo "Cleaning up..."
rm train2017.zip val2017.zip annotations_trainval2017.zip

echo "COCO 2017 filtered subset installation complete!"
echo "Dataset directory: $DATASET_DIR"
echo "Training subset: $DATASET_DIR/train2017_subset"
echo "Validation subset: $DATASET_DIR/val2017_subset"
echo "Filtered annotations: $DATASET_DIR/annotations_subset"
