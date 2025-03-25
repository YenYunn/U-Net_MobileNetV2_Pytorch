import os
import torch
import onnx
import onnxsim


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def export_onnx(model, dummy_input, export_path, input_names, output_names):
    torch.onnx.export(
        model, dummy_input, export_path,
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
    model_path = '../runs/train8/weights/best.pt'
    img_size = (320, 320)
    input_names = ['input_2']
    output_names = ['Identity']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = load_model(model_path, device)

        dummy_input = torch.randn(1, 3, img_size[1], img_size[0]).to(device)

        model_dir = os.path.dirname(model_path)
        export_path = os.path.join(model_dir, 'best.onnx')
        simplified_path = os.path.join(model_dir, 'best_simplified.onnx')

        export_onnx(model, dummy_input, export_path, input_names, output_names)
        simplify_onnx(export_path, simplified_path)

    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
