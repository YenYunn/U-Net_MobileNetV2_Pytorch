import argparse
import os
import re
import csv
import math
import logging
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from dataset import SegDataset
from general import print_args
from losses import combinedLoss
from split_train_val import split_data
from transform import train_transform, val_transform

matplotlib.use('TkAgg')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def plot_all_metrics(scores_dict, save_path='metrics_combined.png'):
    metrics = list(scores_dict.keys())
    num_metrics = len(metrics)

    num_cols = 3
    num_rows = math.ceil(num_metrics / num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

    axes = np.array(axes).flatten()

    for i, (name, scores) in enumerate(scores_dict.items()):
        ax = axes[i]
        ax.plot(range(len(scores['train'])), scores['train'], label=f'train {name}')
        ax.plot(range(len(scores['valid'])), scores['valid'], label=f'val {name}')
        ax.set_title(f'{name} plot')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.legend()

    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_next_train_folder(base_path):
    os.makedirs(base_path, exist_ok=True)

    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    train_dirs = [d for d in existing_dirs if re.fullmatch(r'train\d*', d)]

    if 'train' not in train_dirs:
        new_train_folder = os.path.join(base_path, 'train')
    else:
        train_numbers = [int(re.search(r'\d+', d).group()) if re.search(r'\d+', d) else 0 for d in train_dirs]
        next_train_num = max(train_numbers, default=0) + 1
        new_train_folder = os.path.join(base_path, f'train{next_train_num}')

    # 創建新的資料夾
    os.makedirs(new_train_folder, exist_ok=True)

    return new_train_folder


def main(args):
    # ======================= config ======================== #
    folder = args.data_path
    BATCH_SIZE = args.batch_size
    num_workers = args.num_workers
    num_classes = args.classes
    model_savepath = args.model_savepath
    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights if args.encoder_weights.lower() != 'none' else None
    ACTIVATION = args.activation
    loss_metric = 'combined_loss'

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    decay_rate = args.decay_rate
    decay_steps = args.decay_steps

    # Split dataset into train/val sets
    split_data(args.data_path)

    train_folder = get_next_train_folder(model_savepath)

    continue_training = bool(args.pretrained_weights_path)
    model_path = args.pretrained_weights_path if continue_training else None

    # ============================dataset===============================================================#

    train_set = SegDataset('data/train.txt', folder, train_transform)
    valid_set = SegDataset('data/val.txt', folder, val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # model
    if continue_training:
        if model_path is not None:
            model = torch.load(model_path)
            logger.info('Pretrained weights loaded successfully!')
        else:
            raise ValueError('Error: continue_training is set to 1, but model_path is None!')
    else:
        model = smp.Unet(encoder_name=ENCODER,
                         encoder_weights=ENCODER_WEIGHTS,
                         classes=num_classes,
                         activation=ACTIVATION)
        logger.info('No pretrained weights, training from scratch.')

    loss = combinedLoss([1, 0.8])

    metrics = [smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(), smp.utils.metrics.Dice(),
               smp.utils.metrics.Accuracy(), smp.utils.metrics.Recall(), smp.utils.metrics.Precision(),
               smp.utils.metrics.WarpingError()]

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)])

    train_epoch = smp.utils.train.TrainEpoch(model,
                                             loss=loss,
                                             metrics=metrics,
                                             optimizer=optimizer,
                                             device=DEVICE,
                                             verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(model,
                                             loss=loss,
                                             metrics=metrics,
                                             device=DEVICE,
                                             verbose=True)

    max_score = 0
    losses = {}
    ious = {}

    losses['train'] = []
    losses['valid'] = []
    ious['train'] = []
    ious['valid'] = []
    DICE = {'train': [], 'valid': []}
    RECALL = {'train': [], 'valid': []}
    PRECISION = {'train': [], 'valid': []}
    ACCURACY = {'train': [], 'valid': []}
    WarpingError = {'train': [], 'valid': []}

    # training
    epoch = args.epochs
    weights_folder = os.path.join(train_folder, 'weights')
    os.makedirs(weights_folder, exist_ok=True)

    result_csv_path = os.path.join(train_folder, 'results.csv')

    with open(result_csv_path, 'w', newline='') as dice_file:
        csv_writer = csv.writer(dice_file)
        csv_writer.writerow([
            'Epoch', 'Train Dice', 'Validation Dice', 'Train IoU', 'Validation IoU',
            'Train Accuracy', 'Validation Accuracy', 'Train Precision', 'Validation Precision',
            'Train Recall', 'Validation Recall', 'Train WarpingError', 'Validation WarpingError'
        ])

        for i in range(0, epoch):
            print(f'\nEpoch: {i+1}')
            train_logs = train_epoch.run(train_loader, i)
            valid_logs = valid_epoch.run(valid_loader, i)

            losses.setdefault('train', []).append(train_logs.get(loss_metric, 0))
            losses.setdefault('valid', []).append(valid_logs.get(loss_metric, 0))
            ious.setdefault('train', []).append(train_logs.get('iou_score', 0))
            ious.setdefault('valid', []).append(valid_logs.get('iou_score', 0))

            DICE.setdefault('train', []).append(train_logs.get('dice_score', 0))
            DICE.setdefault('valid', []).append(valid_logs.get('dice_score', 0))

            ACCURACY.setdefault('train', []).append(train_logs.get('accuracy', 0))
            ACCURACY.setdefault('valid', []).append(valid_logs.get('accuracy', 0))

            PRECISION.setdefault('train', []).append(train_logs.get('precision', 0))
            PRECISION.setdefault('valid', []).append(valid_logs.get('precision', 0))

            RECALL.setdefault('train', []).append(train_logs.get('recall', 0))
            RECALL.setdefault('valid', []).append(valid_logs.get('recall', 0))

            WarpingError.setdefault('train', []).append(train_logs.get('warping_error', 0))
            WarpingError.setdefault('valid', []).append(valid_logs.get('warping_error', 0))

            csv_writer.writerow([
                i, train_logs.get('dice_score', 0), valid_logs.get('dice_score', 0), train_logs.get('iou_score', 0),
                valid_logs.get('iou_score', 0),
                train_logs.get('accuracy', 0), valid_logs.get('accuracy', 0), train_logs.get('precision', 0),
                valid_logs.get('precision', 0),
                train_logs.get('recall', 0), valid_logs.get('recall', 0), train_logs.get('warping_error', 0),
                valid_logs.get('warping_error', 0)
            ])

            if max_score < valid_logs.get('iou_score', 0):
                save_model_path = os.path.join(weights_folder, 'best.pt')
                torch.save(model, save_model_path)
                print(f'Model saved! {save_model_path}')

            optimizer.param_groups[0]['lr'] = learning_rate * decay_rate ** (i / decay_steps)

        last_model_path = os.path.join(weights_folder, 'last.pt')
        torch.save(model, last_model_path)
        print(f'Final model saved! f{last_model_path}')

        scores_dict = {
            'loss': losses,
            'iou': ious,
            'dice_score': DICE,
            'precision': PRECISION,
            'recall': RECALL,
            'accuracy': ACCURACY,
            'warping_error': WarpingError
        }

        plot_path = os.path.join(train_folder, 'results.png')
        plot_all_metrics(scores_dict, plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GuideWire Segmentation Training Script')

    # Data & Model Paths
    parser.add_argument('--data_path', type=str, default='./data/images', help='Path to dataset')
    parser.add_argument('--model_savepath', type=str, default='./model', help='Path to save trained models')
    parser.add_argument('--pretrained_weights_path', type=str, default='', help='Path to pretrained weights')

    # Model Hyperparameters
    parser.add_argument('--encoder', type=str, default='mobilenet_v2', help='Encoder backbone')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='Pretrained weights (imagenet or None)')
    parser.add_argument('--activation', type=str, default='sigmoid', help='Activation function')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--classes', type=int, default=1, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--img_size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--decay_steps', type=float, default=1.5, help='Steps for learning rate decay')

    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--num_workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    args = parser.parse_args()

    print_args(vars(args))
    logger.info('start training...')

    main(args)
