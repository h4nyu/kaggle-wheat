import albumentations as A
import typing as t
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(size: int = 1024) -> t.Any:
    return A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(int(size * 0.8), size), height=size, width=size, p=0.5
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.9
                    ),
                ],
                p=0.9,
            ),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
            A.OneOf(
                [A.Blur(blur_limit=3, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.1
            ),
            A.Resize(height=size, width=size, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms(size: int = 1024) -> t.Any:
    return A.Compose(
        [A.Resize(height=size, width=size, p=1.0), ToTensorV2(p=1.0),],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
