import cv2
import os
import onnx
import torch
import onnxsim
from torchvision import transforms


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path, img_size=(320, 320)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    image = cv2.imread(img_path)
    image = cv2.resize(image, img_size)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def export_onnx(model, image_tensor, export_path, input_names, output_names):
    torch.onnx.export(
        model, image_tensor, export_path,
        verbose=True, input_names=input_names, output_names=output_names
    )
    print(f'ONNX model exported to {export_path}')


def simplify_onnx(onnx_path, simplified_path):
    model = onnx.load(onnx_path)
    model_simp, check = onnxsim.simplify(model)
    if check:
        onnx.save(model_simp, simplified_path)
        print(f'Simplified ONNX model saved to {simplified_path}')
    else:
        print('ONNX simplification failed')

def main():
    model_path = '../model/train1/weights/best.pt'
    img_path = '../data/images/0000.jpg'
    export_path = '../model/train1/weights/best.onnx'
    simplified_path = '../model/train1/weights/best_simplified.onnx'
    img_size = (320, 320)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = load_model(model_path, device)
        image_tensor = preprocess_image(img_path, img_size).to(device)

        input_names = ['input_2']
        output_names = ['Identity']

        export_onnx(model, image_tensor, export_path, input_names, output_names)
        simplify_onnx(export_path, simplified_path)
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
