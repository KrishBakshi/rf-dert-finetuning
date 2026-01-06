from rfdetr import RFDETRBase
from rfdetr import RFDETRSegPreview
# model = RFDETRBase(
#     pretrain_weights="./output/checkpoint.pth",
#     image_size=1280,
#     patch_size=12,
#     num_queries=200,
#     group_detr=13,
#     resolution=432,
#     positional_encoding_size=36,
#     num_windows=2,
#     square_resize_div_64=True
# )

model = RFDETRSegPreview(
    pretrain_weights="./output/checkpoint.pth",
    image_size=1280,
    # patch_size=12,
    # num_queries=200,
    # group_detr=13,
    # resolution=432,
    # positional_encoding_size=36,
    # num_windows=2,
    # square_resize_div_64=True
)

detections = model.predict("./dataset/test/18607134-7_page0_crop_000_roi_0.86.png", threshold=0.2)

print(detections)