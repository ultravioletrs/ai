import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def load_model(model_path, class_names):
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(in_features=512, out_features=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def predict(model, image_path, class_names):
    image = preprocess_image(image_path)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    predicted_class = class_names[preds.item()]
    return predicted_class


def show_image_with_prediction(image_path, predicted_class, output_dir):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    
    # Ensure the results directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the output file path
    output_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_path)}")
    
    # Save the image with prediction
    plt.savefig(output_path)
    print(f"Image with prediction saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Load model and make a prediction.")
    parser.add_argument(
        "--model", type=str, required=True, help="path to the model file"
    )
    parser.add_argument(
        "--binary", type=str, required=False, help="path to the binary file from cocos"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="path to the image for prediction"
    )

    args = parser.parse_args()
    if args.binary is not None and args.binary != "":
        if not os.path.isfile(args.binary):
            print(f"Error: Binary file {args.binary} does not exist.")
            return
        try:
            with open(args.binary, "rb") as f:
                data = f.read()
        except Exception as e:
            print(f"Error: Could not read binary file: {e}")
            return

        args.model = "/tmp/model.pth"
        try:
            with open(args.model, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"Error: Could not write model file: {e}")
            return

    model_path = args.model
    image_path = args.image

    class_names = ["Normal", "Viral Pneumonia", "COVID"]

    model = load_model(model_path, class_names)
    predicted_class = predict(model, image_path, class_names)
    print(f"The predicted class for the image is: {predicted_class}")
    
    # Save the image with the prediction to the results directory
    output_dir = "./results"
    show_image_with_prediction(image_path, predicted_class, output_dir)


if __name__ == "__main__":
    main()