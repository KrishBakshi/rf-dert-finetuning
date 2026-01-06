import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np
from rfdetr import RFDETRSegPreview

def visualize_predictions(image_path, detections, output_path, class_names=None, show_masks=True, mask_only=False, instance_segmentation=True):
    """
    Visualize predictions on an image and save to output path.
    
    Args:
        image_path: Path to the input image
        detections: Detections object from model.predict()
        output_path: Path to save the visualization
        class_names: List of class names (optional)
        show_masks: Whether to overlay segmentation masks (default: True)
        mask_only: Show only masks on black background (default: False)
        instance_segmentation: Create instance segmentation visualization (default: True)
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    img_h, img_w = img_array.shape[:2]
    
    num_detections = len(detections.xyxy) if detections.xyxy is not None and len(detections.xyxy) > 0 else 0
    
    # Generate colors for instances
    if num_detections > 0:
        if num_detections <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, num_detections))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, num_detections))
    
    # Create instance segmentation visualization (side-by-side)
    if instance_segmentation and show_masks and detections.mask is not None and num_detections > 0:
        # Create a figure with two subplots: original + instance segmentation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Left: Original image with masks and bboxes
        ax1.imshow(img_array)
        ax1.axis('off')
        ax1.set_title('Original Image with Detections', fontsize=16, pad=10, weight='bold')
        
        # Right: Instance segmentation (colored masks)
        ax2.imshow(img_array)
        ax2.axis('off')
        ax2.set_title('Instance Segmentation', fontsize=16, pad=10, weight='bold')
        
        # Create instance mask overlay for right plot
        instance_overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
        
        for i, (bbox, conf, cls_id, mask) in enumerate(zip(
                detections.xyxy, 
                detections.confidence, 
                detections.class_id,
                detections.mask
            )):
                # Convert boolean mask to float
                if mask.dtype == bool:
                    mask = mask.astype(np.float32)
                elif mask.dtype != np.float32:
                    mask = mask.astype(np.float32)
                    if mask.max() > 1.0:
                        mask = mask / 255.0
                
                # Resize mask to image dimensions if needed
                if mask.shape != (img_h, img_w):
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_uint8, mode='L')
                    mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
                    mask = np.array(mask_pil).astype(np.float32) / 255.0
                    mask = (mask > 0.5).astype(np.float32)
                
                color = colors[i]
                
                # Left plot: semi-transparent overlay
                mask_colored_left = np.zeros((img_h, img_w, 4), dtype=np.float32)
                mask_colored_left[:, :, :3] = color[:3]
                mask_colored_left[:, :, 3] = mask * 0.4
                
                # Right plot: more opaque for instance segmentation
                mask_colored_right = np.zeros((img_h, img_w, 4), dtype=np.float32)
                mask_colored_right[:, :, :3] = color[:3]
                mask_colored_right[:, :, 3] = mask * 0.7  # More opaque for instance view
                
                # Apply to both plots
                instance_overlay = np.maximum(instance_overlay, mask_colored_right)
                
                # Draw on left plot
                ax1.imshow(mask_colored_left)
                
                # Draw bounding box on left plot
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax1.add_patch(rect)
                
                # Add instance label
                class_name = class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else f'Class {int(cls_id)}'
                label = f'#{i+1}: {class_name} {conf:.2f}'
                ax1.text(
                    x1, y1 - 5, label,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.9),
                    fontsize=9, color='white', weight='bold'
                )
                
                # Draw bounding box on right plot (instance segmentation)
                rect2 = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2.5, edgecolor='white', facecolor='none', linestyle='--'
                )
                ax2.add_patch(rect2)
                
                # Add instance number in center of bbox
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax2.text(
                    center_x, center_y, f'{i+1}',
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor=color, alpha=0.9, edgecolor='white', linewidth=2),
                    fontsize=14, color='white', weight='bold'
                )
        
        # Apply instance overlay to right plot
        ax2.imshow(instance_overlay)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved instance segmentation to: {output_path}")
        return
    
    # Fallback to single plot if instance_segmentation is False or no masks
    # Create figure for single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    if mask_only:
        # Show only masks on black background for debugging
        ax.imshow(np.zeros_like(img_array))
    else:
        ax.imshow(img_array)
    ax.axis('off')
    
    if num_detections > 0:
        # Draw segmentation masks if available
        if show_masks and detections.mask is not None:
            print(f"  Drawing {len(detections.mask)} masks...")
            # Create a combined mask overlay
            mask_overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
            
            for i, (bbox, conf, cls_id, mask) in enumerate(zip(
                detections.xyxy, 
                detections.confidence, 
                detections.class_id,
                detections.mask
            )):
                # Convert boolean mask to float (True -> 1.0, False -> 0.0)
                if mask.dtype == bool:
                    mask = mask.astype(np.float32)
                elif mask.dtype != np.float32:
                    mask = mask.astype(np.float32)
                    # Normalize if values are > 1.0
                    if mask.max() > 1.0:
                        mask = mask / 255.0
                
                # Resize mask to image dimensions if needed
                # Important: Use NEAREST neighbor to preserve mask boundaries
                if mask.shape != (img_h, img_w):
                    # Convert to uint8 for PIL (0-255 range)
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_uint8, mode='L')  # Grayscale
                    # Use NEAREST to preserve sharp mask edges (not bilinear which would blur)
                    mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
                    mask = np.array(mask_pil).astype(np.float32) / 255.0
                    # Ensure binary mask (threshold at 0.5 to handle any interpolation artifacts)
                    mask = (mask > 0.5).astype(np.float32)
                
                # Apply color to mask
                color = colors[i]
                mask_colored = np.zeros((img_h, img_w, 4), dtype=np.float32)
                mask_colored[:, :, :3] = color[:3]
                # Only apply color where mask is True (not a filled rectangle)
                mask_colored[:, :, 3] = mask * 0.6  # 60% opacity for better visibility of actual mask shape
                
                # Combine masks (use maximum to overlay multiple masks)
                mask_overlay = np.maximum(mask_overlay, mask_colored)
            
            # Overlay masks on image (after the base image)
            if mask_overlay.max() > 0:
                ax.imshow(mask_overlay, alpha=1.0)  # Alpha is already in mask_overlay
                print(f"  Masks overlaid successfully (max value: {mask_overlay.max():.3f})")
            else:
                print(f"  Warning: mask_overlay is empty!")
        
        # Draw bounding boxes (skip if mask_only mode)
        if not mask_only:
            for i, (bbox, conf, cls_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                x1, y1, x2, y2 = bbox
                
                # Create rectangle with color matching mask
                color = colors[i] if show_masks and detections.mask is not None else 'red'
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                class_name = class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else f'Class {int(cls_id)}'
                label = f'{class_name}: {conf:.2f}'
                ax.text(
                    x1, y1 - 5, label,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                    fontsize=10, color='white', weight='bold'
                )
    
    # Add title with detection count
    title = f'{Path(image_path).name}\nDetections: {num_detections}'
    if show_masks and detections.mask is not None:
        title += ' (with masks)'
    ax.set_title(title, fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved visualization to: {output_path}")

def predict_and_visualize(
    model,
    image_path_or_dir,
    output_dir="./predictions",
    confidence_threshold=0.2,
    class_names=None,
    show_masks=True
):
    """
    Run predictions on images and save visualizations.
    
    Args:
        model: Loaded RFDETRBase model
        image_path_or_dir: Path to single image or directory of images
        output_dir: Directory to save prediction visualizations
        confidence_threshold: Confidence threshold for detections
        class_names: List of class names (optional)
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
    
    print(f"Found {len(image_paths)} image(s) to process")
    
    # Process each image
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        
        # Run prediction
        detections = model.predict(image_path, threshold=confidence_threshold)
        
        # Print detection info
        num_detections = len(detections.xyxy) if detections.xyxy is not None and len(detections.xyxy) > 0 else 0
        print(f"  Detections: {num_detections}")
        if num_detections > 0:
            print(f"  Confidence range: {detections.confidence.min():.3f} - {detections.confidence.max():.3f}")
        
        # Create output filename
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_prediction.png")
        
        # Visualize and save
        visualize_predictions(image_path, detections, output_path, class_names, show_masks=show_masks)
    
    print(f"\n✓ All predictions saved to: {output_dir}")

if __name__ == "__main__":
    # Load model with correct configuration (using RFDETRSegPreview for segmentation)
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
    
    # Run predictions on test directory
    predict_and_visualize(
        model=model,
        image_path_or_dir="./dataset/test",  # Can also specify a single image path
        output_dir="./predictions",
        confidence_threshold=0.2,
        class_names=class_names,
        show_masks=True  # Set to False to show only bounding boxes
    )

