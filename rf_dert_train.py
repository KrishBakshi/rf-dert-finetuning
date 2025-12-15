from rfdetr import RFDETRSegPreview

if __name__ == "__main__":
    model = RFDETRSegPreview()

    model.train(
        dataset_dir="./dataset/",
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="./output/",
        device="cuda",
        tensorboard=True,
        image_size=1280,
        # resume= "./output/checkpoint.pth"
    )