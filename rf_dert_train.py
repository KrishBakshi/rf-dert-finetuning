import warnings
warnings.filterwarnings("ignore")
from rfdetr import RFDETRSeg2XLarge

#The follwing is for x1 a100 80GB SMX/SMX4 GPU
if __name__ == "__main__":
    model = RFDETRSeg2XLarge()

    model.train(
        dataset_dir="./dataset/",
        epochs=100,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir="./output/",
        device="cuda",
        tensorboard=True,
        resolution=1272,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.005,
        # resume= "./output/checkpoint.pth"
    )