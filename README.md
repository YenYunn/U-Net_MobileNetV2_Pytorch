# UNet-MobileNetV2-Pytorch

## 🚀How to run
Glone this repo, and create an environment and run:

`pip install -r requirements.txt`

---
## 📂 Dataset Format
```
data/
│── images/
│   │── xxxx.jpg
│   │── xxxx_mask.png
│   │── xxxx.jpg
│   │── xxxx_mask.png
│   │── ...
│   │── ...
│   │── ...
│   │── label_names.txt
```
---

## 🎯 Training

```shell script
> python train.py -h

usage: train.py [-h] [--data_path DATA_PATH] [--model_savepath MODEL_SAVEPATH]
                [--pretrained_weights_path PRETRAINED_WEIGHTS_PATH]
                [--encoder ENCODER] [--encoder_weights ENCODER_WEIGHTS]
                [--activation ACTIVATION] [--epochs EPOCHS] [--classes CLASSES]
                [--batch_size BATCH_SIZE] [--img_size IMG_SIZE]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                [--decay_rate DECAY_RATE] [--decay_steps DECAY_STEPS]
                [--device DEVICE] [--num_workers NUM_WORKERS]

Train the UNet on images and target masks

optional arguments:
   -h, --help                              show this help message and exit
  --data_path DATA_PATH                    Path to dataset (default: ./data/images)
  --model_savepath MODEL_SAVEPATH          Path to save trained models (default: ./model)
  --pretrained_weights_path PRETRAINED_WEIGHTS_PATH  
                                           Path to pretrained weights (default: '')
  --encoder ENCODER                        Encoder backbone (default: mobilenet_v2)
  --encoder_weights ENCODER_WEIGHTS        Pretrained weights (imagenet or None) (default: imagenet)
  --activation ACTIVATION                  Activation function (default: sigmoid)
  --epochs EPOCHS                          Number of epochs (default: 100)
  --classes CLASSES                        Number of classes (default: 1)
  --batch_size BATCH_SIZE                  Batch size (-1 for autobatch) (default: 16)
  --img_size IMG_SIZE                      Train/val image size (pixels) (default: 320)
  --learning_rate LEARNING_RATE            Initial learning rate (default: 0.001)
  --weight_decay WEIGHT_DECAY              Weight decay (default: 0.0005)
  --decay_rate DECAY_RATE                  Learning rate decay factor (default: 0.9)
  --decay_steps DECAY_STEPS                Steps for learning rate decay (default: 1.5)
  --device DEVICE                          Training device (e.g., cuda:0 or cpu) (default: cuda:0)
  --num_workers NUM_WORKERS                Number of dataloader workers (default: 0)
```
---
## 🔍Prediction

To predict a single image and save it:

`python predict.py --model_path model.pt/pth --input image.jpg/video.mp4 --device 0 --save_mask`


---
## 📚Reference
https://github.com/milesial/Pytorch-UNet
