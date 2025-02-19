import cv2
import torch
import argparse
import numpy as np
from torchvision import transforms

INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)


def predict_mask(image, model, device):
    image = preprocess_image(image).to(device)
    with torch.no_grad():
        pr_mask = model.predict(image)
    return pr_mask.squeeze().cpu().numpy().round()


def visualize_results(image, mask):
    pred_mask_show = (mask * 255).astype(np.uint8)
    mask_rgb = cv2.merge([pred_mask_show, np.zeros_like(pred_mask_show), np.zeros_like(pred_mask_show)])

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

    cv2.imshow('Prediction Mask', pred_mask_show)
    cv2.imshow('Overlay Result', blended)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)


def save_mask(mask, output_path):
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)


def main():
    parser = argparse.ArgumentParser(description='Image Segmentation with a Trained Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--save_mask', action='store_true', help='Flag to save the predicted mask')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == '0' else 'cpu')
    model = load_model(args.model_path, device)

    image = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), -1)
    image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

    mask = predict_mask(image, model, device)
    visualize_results(image, mask)

    if args.save_mask:
        output_path = os.path.splitext(args.image_path)[0] + '_mask.png'
        save_mask(mask, output_path)
        print(f'Mask saved at: {output_path}')


if __name__ == '__main__':
    main()
