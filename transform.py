import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

train_transform = albu.Compose([albu.Resize(320, 320),
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.OneOf([
                                    albu.RandomBrightness(limit=0.2, p=1),
                                    albu.RandomContrast(limit=0.2, p=1),
                                    albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=0,
                                                            p=1)
                                ], p=0.5),
                                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensor()
                                ])

val_transform = albu.Compose([albu.Resize(320, 320),
                              albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensor()
                              ])
