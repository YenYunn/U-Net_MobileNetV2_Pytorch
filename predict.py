import os
import cv2
import torch
import argparse
import numpy as np
from torchvision import transforms


INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320
DEFAULT_X_START = 640
DEFAULT_CROP_WIDTH = 640

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image):
    _, width = image.shape[:2]
    if width >= DEFAULT_X_START + DEFAULT_CROP_WIDTH:
        crop_image_original = image[:, DEFAULT_X_START:DEFAULT_X_START + DEFAULT_CROP_WIDTH]
    else:
        crop_image_original = image

    img = cv2.cvtColor(crop_image_original, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
    transform = transforms.Compose([transforms.ToTensor()])
    input_img = transform(img_resized).unsqueeze(0)

    return crop_image_original, input_img


def predict_mask(image, model, device):
    crop_image_original, input_img = preprocess_image(image)
    input_img = input_img.to(device)

    start_inference = cv2.getTickCount()  # 記錄推論開始時間
    with torch.no_grad():
        pr_mask = model.predict(input_img)
    end_inference = cv2.getTickCount()  # 記錄推論結束時間

    inference_time = (end_inference - start_inference) / cv2.getTickFrequency() * 1000  # 轉換為 ms

    pr_mask_img = pr_mask.squeeze().cpu().numpy().round()
    pr_mask_img = cv2.resize(pr_mask_img, (crop_image_original.shape[1], crop_image_original.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    return crop_image_original, pr_mask_img, inference_time


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

    return pred_mask_show


def show_video_results(frame, mask):
    pred_mask_show = (mask * 255).astype(np.uint8)
    mask_rgb = cv2.merge([pred_mask_show, np.zeros_like(pred_mask_show), np.zeros_like(pred_mask_show)])
    blended = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)
    return pred_mask_show, blended


def process_folder(folder_path, model, device):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]
    image_files.sort()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if image is None:
            continue
        crop_img, mask, inference_time = predict_mask(image, model, device)
        visualize_results(crop_img, mask)
        print(f'Processed {image_file} - Inference Time: {inference_time:.2f} ms')

def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f'Error: Cannot open video {video_path}')
        return

    frame_times = []  # 記錄每一幀的總執行時間
    inference_times = []  # 記錄單獨推論時間
    frame_count = 0

    while cap.isOpened():
        start_time = cv2.getTickCount()  # 記錄開始時間

        ret, frame = cap.read()
        if not ret:
            break

        crop_frame, mask, inference_time = predict_mask(frame, model, device)
        pred_mask, blended = show_video_results(crop_frame, mask)

        cv2.imshow('Prediction Mask', pred_mask)
        cv2.imshow('Overlay Result', blended)
        cv2.imshow('Original Image', crop_frame)

        end_time = cv2.getTickCount()  # 記錄結束時間
        frame_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # 轉換為 ms

        frame_times.append(frame_time)
        inference_times.append(inference_time)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 計算平均數據
    avg_frame_time = np.mean(frame_times) if frame_times else 0
    avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0

    print(f'\n[Performance Summary]')
    print(f'Average Total Execution Time per Frame: {avg_frame_time:.2f} ms')
    print(f'Average FPS: {avg_fps:.2f}')
    print(f'Average Inference Time per Frame: {avg_inference_time:.2f} ms')


def main():
    parser = argparse.ArgumentParser(description='Image Segmentation with a Trained Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image/video/stream')
    parser.add_argument('--device', type=str, default='0', help='Device to run the model')
    parser.add_argument('--save_mask', action='store_true', help='Flag to save the predicted mask')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == '0' else 'cpu')
    print(f'device: {device}')
    model = load_model(args.model_path, device)

    if os.path.isdir(args.input):
        process_folder(args.input, model, device)
    elif args.input.isdigit() or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_source = int(args.input) if args.input.isdigit() else args.input
        process_video(video_source, model, device)
    else:
        image = cv2.imdecode(np.fromfile(args.input, dtype=np.uint8), -1)
        crop_img, mask, inference_time = predict_mask(image, model, device)
        mask = visualize_results(crop_img, mask)

        print(f'Inference Time: {inference_time:.2f} ms')

        if args.save_mask:
            output_path = os.path.splitext(args.input)[0] + '_mask.png'
            cv2.imencode('.png', mask)[1].tofile(output_path)
            print(f'Mask saved at: {output_path}')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
