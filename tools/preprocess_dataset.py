import os
import cv2


def rename_mask_in_directory(directory: str, suffix: str = '_mask', extension: str = '.png'):
    if not os.path.isdir(directory):
        print(f'Error：{directory} does not exit.')
        return

    try:
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                old_path = os.path.join(directory, filename)
                name, ext = os.path.splitext(filename)

                if not name.endswith(suffix):
                    new_filename = f'{name}{suffix}{ext}'
                    new_path = os.path.join(directory, new_filename)

                    os.rename(old_path, new_path)
                    print(f'Renamed: {filename} -> {new_filename}')

        print('Batch renaming completed!')
    except Exception as e:
        print(f'An error occurred: {e}')


def crop_and_save_all(input_base_folder, output_folder, x_start=640, crop_width=640):
    os.makedirs(output_folder, exist_ok=True)

    for subfolder, file_extension in [('images', '.jpg'), ('mask', '.png')]:
        input_folder = os.path.join(input_base_folder, subfolder)

        if not os.path.exists(input_folder):
            print(f'Skipping missing folder: {input_folder}')
            continue

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(file_extension):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                image = cv2.imread(input_path)
                if image is None:
                    print(f'Failed to read image: {input_path}')
                    continue

                if image.shape[1] < x_start + crop_width:
                    print(f'Image width too small: {input_path}')
                    continue

                cropped_image = image[:, x_start:x_start + crop_width]
                if subfolder == 'mask':
                    cropped_image[cropped_image == 255] = 1
                cv2.imwrite(output_path, cropped_image)
                print(f'Cropped and saved: {output_path}')
                

if __name__ == '__main__':
    mask_directory = '../data/datasets/mask'

    # rename_mask_in_directory(mask_directory)
    crop_and_save_all('../data/datasets', '../data/images')
