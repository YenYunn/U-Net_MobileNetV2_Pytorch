import os
from sklearn.model_selection import train_test_split


def split_data(root_path):
    filenames = set()

    for root, dirs, files in os.walk(root_path):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext.lower() == '.jpg' and not filename.endswith('k.png'):
                filenames.add(name)

    filenames = list(filenames)
    X_train, X_val = train_test_split(filenames, test_size=0.1, random_state=29)

    output_dir = os.path.dirname(root_path)

    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')

    with open(train_path, mode='w', encoding='utf-8') as train_file:
        for train_name in X_train:
            if train_name != 'label_names':
                train_file.write(train_name + '\n')

    with open(val_path, mode='w', encoding='utf-8') as val_file:
        for val_name in X_val:
            if val_name != 'label_names':
                val_file.write(val_name + '\n')
