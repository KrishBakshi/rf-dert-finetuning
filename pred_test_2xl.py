import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from rfdetr import RFDETRSeg2XLarge


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise ValueError(f"Invalid input path: {path}")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    images: list[Path] = []
    for ext in exts:
        images.extend(path.glob(ext))

    return sorted(images)


def visualize_prediction(
    image_path: Path,
    detections,
    output_path: Path,
    class_names: list[str],
    show_masks: bool = True,
) -> None:
    image = Image.open(image_path)
    image_np = np.array(image)
    img_h, img_w = image_np.shape[:2]

    num_detections = (
        len(detections.xyxy)
        if detections.xyxy is not None and len(detections.xyxy) > 0
        else 0
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image_np)
    ax.axis("off")

    if num_detections > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_detections, 1)))

        if show_masks and detections.mask is not None:
            mask_overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
            for i, mask in enumerate(detections.mask):
                if mask.dtype == bool:
                    mask = mask.astype(np.float32)
                elif mask.dtype != np.float32:
                    mask = mask.astype(np.float32)
                    if mask.max() > 1.0:
                        mask = mask / 255.0

                if mask.shape != (img_h, img_w):
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                    mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
                    mask = (np.array(mask_img).astype(np.float32) / 255.0 > 0.5).astype(
                        np.float32
                    )

                color = colors[i]
                mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.float32)
                mask_rgba[:, :, :3] = color[:3]
                mask_rgba[:, :, 3] = mask * 0.5
                mask_overlay = np.maximum(mask_overlay, mask_rgba)

            ax.imshow(mask_overlay, alpha=1.0)

        for i, (bbox, conf, cls_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            class_idx = int(cls_id)
            class_name = (
                class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            )
            label = f"{class_name}: {conf:.2f}"
            ax.text(
                x1,
                max(y1 - 5, 0),
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.85),
            )

    ax.set_title(f"{image_path.name} | detections={num_detections}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_predictions(
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    confidence: float,
    image_size: int,
    device: str,
    class_names: list[str],
    show_masks: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RF-DETR Seg 2XL model...")
    model = RFDETRSeg2XLarge(
        pretrain_weights=str(checkpoint),
        image_size=image_size,
        device=device,
    )
    print("Model loaded.")

    image_paths = collect_images(input_path)
    print(f"Found {len(image_paths)} image(s) in {input_path}")

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Predicting {image_path.name}")
        detections = model.predict(str(image_path), threshold=confidence)

        out_file = output_dir / f"{image_path.stem}_prediction.png"
        visualize_prediction(
            image_path=image_path,
            detections=detections,
            output_path=out_file,
            class_names=class_names,
            show_masks=show_masks,
        )
        print(f"Saved: {out_file}")

    print(f"\nDone. Predictions written to {output_dir}")


if __name__ == "__main__":
    run_predictions(
        checkpoint=Path("./output/./output/checkpoint_best_total.pth"),
        input_path=Path("./dataset/test"),
        output_dir=Path("./predictions/test_2xl"),
        confidence=0.2,
        image_size=1272,
        device="cuda",
        class_names=["room"],
        show_masks=True,
    )
