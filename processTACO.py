import os
import json
import cv2
from tqdm import tqdm

def build_taco_classification_dataset(taco_json_path, image_root_dir, output_root_dir, min_crop_size=64):
    """
    Parses TACO COCO annotations, crops individual items, and maps them to 4 target classes.
    """
    # Setup target directories
    target_classes = ['dry', 'wet', 'recyclable', 'e-waste']
    for cls in target_classes:
        os.makedirs(os.path.join(output_root_dir, cls), exist_ok=True)

    # Define the TACO -> EcoSort mapping
    # TACO has 60 categories. We map them using keywords or supercategories.
    recyclable_keywords = ['bottle', 'can', 'carton', 'paper', 'plastic', 'glass', 'metal', 'cup', 'straw']
    dry_keywords = ['cigarette', 'styrofoam', 'unlabeled litter', 'blister pack', 'rope', 'shoe']
    wet_keywords = ['food', 'apple', 'orange', 'peel']
    ewaste_keywords = ['battery']

    with open(taco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create mapping dictionaries
    image_dict = {img['id']: img for img in coco_data['images']}
    category_map = {}

    for cat in coco_data['categories']:
        cat_name = cat['name'].lower()
        super_cat = cat['supercategory'].lower()

        # Determine the target class
        target = 'dry'  # Default fallback
        search_string = f"{cat_name} {super_cat}"

        if any(kw in search_string for kw in recyclable_keywords):
            target = 'recyclable'
        elif any(kw in search_string for kw in wet_keywords):
            target = 'wet'
        elif any(kw in search_string for kw in ewaste_keywords):
            target = 'e-waste'

        category_map[cat['id']] = target

    # process annotations and crop
    valid_crops = 0
    skipped_small = 0

    print("Cropping TACO annotations...")
    for ann in tqdm(coco_data['annotations']):
        image_info = image_dict.get(ann['image_id'])
        if not image_info:
            continue

        # COCO bbox format: [x_top_left, y_top_left, width, height]
        x, y, w, h = [int(v) for v in ann['bbox']]

        # Filter out tiny crops that will pixelate during 224x224 upscaling
        if w < min_crop_size or h < min_crop_size:
            skipped_small += 1
            continue

        # Load the source image
        img_path = os.path.join(image_root_dir, image_info['file_name'])
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Ensure that the bounding box stays within image dimensions
        img_h, img_w = img.shape[:2]
        x_start, y_start = max(0, x), max(0, y)
        x_end, y_end = min(img_w, x + w), min(img_h, y + h)

        # Slice the numpy array
        crop = img[y_start:y_end, x_start:x_end]

        # Skip if the boundary check resulted in an empty array
        if crop.size == 0:
            continue

        # Save to the mapped class directory
        target_class = category_map.get(ann['category_id'], 'dry')
        crop_filename = f"taco_{ann['image_id']}_{ann['id']}.jpg"
        save_path = os.path.join(output_root_dir, target_class, crop_filename)

        cv2.imwrite(save_path, crop)
        valid_crops += 1

    print(f"\nProcessing Complete.")
    print(f"Valid crops saved: {valid_crops}")
    print(f"Crops skipped (under {min_crop_size}x{min_crop_size}): {skipped_small}")

# Execute above function
build_taco_classification_dataset(
   taco_json_path='./TACO/annotations.json',
   image_root_dir='./TACO',
   output_root_dir='./EcoSort_Unified_Dataset',
   min_crop_size=64
)