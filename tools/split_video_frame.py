import os
import re
import cv2

DEFAULT_X_START = 640
DEFAULT_CROP_WIDTH = 640


def crop_frame(frame):
    return frame[:, DEFAULT_X_START:DEFAULT_X_START + DEFAULT_CROP_WIDTH]


def get_prefix(video_name):
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', video_name)
    if match:
        return f'{match.group(1)}{match.group(2)}{match.group(3)}'
    return 'unknown'

def process_video(video_path, frame_save_dir, frame_interval=3):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(frame_save_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    prefix = get_prefix(video_name)
    count = 0
    save_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # cropped_frame = crop_frame(frame)

        if count % frame_interval == 0:
            frame_save_path = os.path.join(frame_save_dir, f'{prefix}_{save_count:04d}.jpg')
            print(f'Saving frame to: {frame_save_path}')
            cv2.imwrite(frame_save_path, frame)
            save_count += 1

        count += 1

    cap.release()


def main():
    videos = [
        '../media/2025-03-11 10-44-57.mkv',
    ]

    frame_save_dir = '../data/rawframes'
    frame_interval = 5

    for video_path in videos:
        process_video(video_path, frame_save_dir, frame_interval)
        print(f'Split video: {video_path} done!')

    print('All done!')


if __name__ == '__main__':
    main()
