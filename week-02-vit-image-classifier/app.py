from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import argparse


def load_image(path=None, url=None):
    if path:
        return Image.open(path).convert("RGB")
    elif url:
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")
    else:
        raise ValueError("Either path or url must be provided.")


def classify(image):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]

    return label


def main():
    parser = argparse.ArgumentParser(description="ViT Image Classifier")
    parser.add_argument("--image", help="Local image path")
    parser.add_argument("--url", help="Image URL")
    args = parser.parse_args()

    if not args.image and not args.url:
        print("Usage:")
        print("  python app.py --image cat.jpg")
        print("  python app.py --url https://example.com/cat.jpg")
        return

    img = load_image(path=args.image, url=args.url)
    label = classify(img)

    print(f"\nPredicted class → {label}\n")


if __name__ == "__main__":
    main()
