from functools import partial

from ..common.xworld_loader_lsj import dataloader
from ..eva2_o365_to_coco.cascade_mask_rcnn_vitdet_b_100ep import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

from detectron2.data.datasets import register_coco_instances

import os

# Get environment variables
WORKDIR_USER = os.getenv("WORKDIR_USER")

for split_type in ["train", "val"]:
    register_coco_instances(
        f"xworld_{split_type}",
        {},
        f"{WORKDIR_USER}/ava-challenge/EVA/EVA-02/det/xworld/{split_type}_xworld.json",
        f"{WORKDIR_USER}/ava-challenge/EVA/EVA-02/det/xworld/{split_type}_xworld",
    )

from detectron2.config import LazyCall as L
from fvcore.common.param_scheduler import *
from detectron2.solver import WarmupParamScheduler

# number of classes is 8
model.roi_heads.num_classes = 8

model.backbone.net.img_size = 1536
model.backbone.square_pad = 1536
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4 * 2 / 3
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.3

# 2, 5, 8, 11, 14, 17, 20, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 2))
    + list(range(3, 5))
    + list(range(6, 8))
    + list(range(9, 11))
    + list(range(12, 14))
    + list(range(15, 17))
    + list(range(18, 20))
    + list(range(21, 23))
)

optimizer.lr = 4e-5
# optimizer.lr = 1e-5
# optimizer.lr = 2e-5
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24
)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = 25000
# train.max_iter = 12500

train.model_ema.enabled = True
train.model_ema.device = "cuda"
train.model_ema.decay = 0.9999

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=1,
    ),
    warmup_length=0.01,
    warmup_factor=0.001,
)

dataloader.test.num_workers = 0
dataloader.train.total_batch_size = 8
# dataloader.train.total_batch_size = 16

train.eval_period = 2500
train.checkpointer.period = 2500
# train.eval_period = 1250
# train.checkpointer.period = 1250
