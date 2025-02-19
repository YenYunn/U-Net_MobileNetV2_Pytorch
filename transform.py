import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

train_transform = albu.Compose([albu.Resize(320, 320),
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                # albu.CenterCrop(width=289, height=289, p=0.1),
                                # albu.RandomBrightness(p=0.1),
                                # albu.OneOf([albu.RandomContrast(p=1),
                                #             albu.HueSaturationValue(p=1)],
                                #            p=0.5),
                                # albu.Normalize(mean=MEAN, std=STD),
                                ToTensor()
                                ])

val_transform = albu.Compose([albu.Resize(320, 320),
                              ToTensor()
                              ])
