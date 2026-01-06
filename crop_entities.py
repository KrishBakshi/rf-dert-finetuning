import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from rfdetr import RFDETRSegPreview

def crop_entity_with_mask(image_array, mask, bbox=None, padding=10):
    """
    Crop an entity from an image using its segmentation mask.
    
    Args:
        image_array: Original image as numpy array
        mask: Binary mask (boolean or float array)
        bbox: Optional bounding box [x1, y1, x2, y2] for cropping region
        padding: Padding around the bbox in pixels
    
    Returns:
        Cropped image as PIL Image, or None if mask is invalid
    """
    # Convert mask to boolean if needed
    if mask.dtype != bool:
        mask = mask > 0.5
    
    # Get mask bounds
    mask_coords = np.where(mask)
    if len(mask_coords[0]) == 0:
        return None  # Empty mask
    
    min_y, max_y = mask_coords[0].min(), mask_coords[0].max()
    min_x, max_x = mask_coords[1].min(), mask_coords[1].max()
    
    # Use bbox if provided, otherwise use mask bounds
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    else:
        x1, y1, x2, y2 = min_x, min_y, max_x, max_y
    
    # Add padding
    img_h, img_w = image_array.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)
    
    # Crop image region
    cropped_image = image_array[y1:y2, x1:x2]
    
    # Crop mask to same region
    cropped_mask = mask[y1:y2, x1:x2]
    
    # Create RGBA image with transparency
    if len(cropped_image.shape) == 2:  # Grayscale
        cropped_rgba = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        cropped_rgba[:, :, 0] = cropped_image
        cropped_rgba[:, :, 1] = cropped_image
        cropped_rgba[:, :, 2] = cropped_image
    else:  # RGB
        cropped_rgba = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        cropped_rgba[:, :, :3] = cropped_image
    
    # Set alpha channel based on mask
    cropped_rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(cropped_rgba, 'RGBA')
    
    return pil_image

def crop_entities_from_image(
    model,
    image_path,
    output_dir,
    confidence_threshold=0.2,
    class_names=None,
    padding=10,
    save_format='png'
):
    """
    Crop all detected entities from an image using their segmentation masks.
    
    Args:
        model: Loaded RFDETRSegPreview model
        image_path: Path to input image
        output_dir: Directory to save cropped entities
        confidence_threshold: Confidence threshold for detections
        class_names: List of class names (optional)
        padding: Padding around each entity in pixels
        save_format: Image format to save ('png' or 'jpg')
    
    Returns:
        List of paths to saved cropped images
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    img_h, img_w = img_array.shape[:2]
    
    # Run prediction
    detections = model.predict(image_path, threshold=confidence_threshold)
    
    num_detections = len(detections.xyxy) if detections.xyxy is not None and len(detections.xyxy) > 0 else 0
    
    if num_detections == 0:
        print(f"  No detections found in {Path(image_path).name}")
        return []
    
    if detections.mask is None:
        print(f"  Warning: No masks available for {Path(image_path).name}")
        return []
    
    # Create output directory for this image
    image_name = Path(image_path).stem
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    saved_paths = []
    
    print(f"  Found {num_detections} detection(s)")
    
    # Crop each detected entity
    for i, (bbox, conf, cls_id, mask) in enumerate(zip(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
        detections.mask
    )):
        # Convert boolean mask to float if needed
        if mask.dtype == bool:
            mask_float = mask.astype(np.float32)
        else:
            mask_float = mask.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0
            mask_float = (mask_float > 0.5).astype(np.float32)
        
        # Resize mask to image dimensions if needed
        if mask.shape != (img_h, img_w):
            from PIL import Image as PILImage
            mask_uint8 = (mask_float * 255).astype(np.uint8)
            mask_pil = PILImage.fromarray(mask_uint8, mode='L')
            mask_pil = mask_pil.resize((img_w, img_h), PILImage.NEAREST)
            mask_float = np.array(mask_pil).astype(np.float32) / 255.0
            mask_float = (mask_float > 0.5).astype(np.float32)
        
        # Convert back to boolean for cropping
        mask_bool = mask_float.astype(bool)
        
        # Crop entity
        cropped_entity = crop_entity_with_mask(img_array, mask_bool, bbox, padding=padding)
        
        if cropped_entity is None:
            print(f"    Skipping entity {i+1}: empty mask")
            continue
        
        # Create filename
        class_name = class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else f'class_{int(cls_id)}'
        filename = f"entity_{i+1:03d}_{class_name}_conf{conf:.3f}.{save_format}"
        output_path = os.path.join(image_output_dir, filename)
        
        # Save cropped entity
        if save_format.lower() == 'jpg' or save_format.lower() == 'jpeg':
            # Convert RGBA to RGB for JPEG
            rgb_image = Image.new('RGB', cropped_entity.size, (255, 255, 255))
            rgb_image.paste(cropped_entity, mask=cropped_entity.split()[3])  # Use alpha channel as mask
            rgb_image.save(output_path, 'JPEG', quality=95)
        else:
            cropped_entity.save(output_path, 'PNG')
        
        saved_paths.append(output_path)
        print(f"    Saved entity {i+1}: {filename}")
    
    return saved_paths

def crop_entities_from_directory(
    model,
    image_path_or_dir,
    output_dir="./cropped_entities",
    confidence_threshold=0.2,
    class_names=None,
    padding=10,
    save_format='png'
):
    """
    Crop all detected entities from images in a directory.
    
    Args:
        model: Loaded RFDETRSegPreview model
        image_path_or_dir: Path to single image or directory of images
        output_dir: Directory to save cropped entities
        confidence_threshold: Confidence threshold for detections
        class_names: List of class names (optional)
        padding: Padding around each entity in pixels
        save_format: Image format to save ('png' or 'jpg')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    if os.path.isfile(image_path_or_dir):
        image_paths = [image_path_or_dir]
    elif os.path.isdir(image_path_or_dir):
        # Get all image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_path_or_dir, ext)))
        image_paths.sort()
    else:
        raise ValueError(f"Invalid path: {image_path_or_dir}")
    
    print(f"Found {len(image_paths)} image(s) to process\n")
    
    total_entities = 0
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing: {Path(image_path).name}")
        
        saved_paths = crop_entities_from_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            class_names=class_names,
            padding=padding,
            save_format=save_format
        )
        
        total_entities += len(saved_paths)
        print()
    
    print(f"✓ Cropped {total_entities} entity/entities from {len(image_paths)} image(s)")
    print(f"✓ Saved to: {output_dir}")

if __name__ == "__main__":
    # Load model with correct configuration
    print("Loading model...")
    model = RFDETRSegPreview(
        pretrain_weights="./output/checkpoint.pth",
        image_size=1280,
        patch_size=12,
        num_queries=200,
        group_detr=13,
        resolution=432,
        positional_encoding_size=36,
        num_windows=2,
        square_resize_div_64=True
    )
    print("Model loaded successfully!\n")
    
    # Class names (from checkpoint args: class_names=['room'])
    class_names = ['room']
    
    # Crop entities from test directory
    crop_entities_from_directory(
        model=model,
        image_path_or_dir="./dataset/test",  # Can also specify a single image path
        output_dir="./cropped_entities",
        confidence_threshold=0.2,
        class_names=class_names,
        padding=10,  # Padding around each entity in pixels
        save_format='png'  # 'png' preserves transparency, 'jpg' for smaller files
    )

