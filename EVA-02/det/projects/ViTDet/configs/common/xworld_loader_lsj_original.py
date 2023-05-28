import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

# Data using LSJ
image_size_w = 1920
image_size_h = 1080
data_loader = model_zoo.get_config("common/data/xworld.py").dataloader
data_loader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size_h, target_width=image_size_w
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size_h, image_size_w), pad=False),
]
data_loader.train.mapper.image_format = "RGB"
data_loader.train.total_batch_size = 64
# recompute boxes due to cropping
data_loader.train.mapper.recompute_boxes = True

data_loader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size_h, max_size=image_size_h),
]
