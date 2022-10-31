# from mmdet.datasets.objaverse import ObjaverseDummyDataset, ObjaverseAugment
# from mmdet.datasets.objaverse import ObjaverseAugment

_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='ObjaverseDummyDataset', ignore_passthrough=True),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
#     dict(
#         type='Resize',
#         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                    (1333, 768), (1333, 800)],
#         multiscale_mode='value',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

# data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.001,
        dataset=dict(
            type="ObjaverseDummyDataset",
            # type="LVISV1Dataset",
            ignore_passthrough=False,
            ann_file='annotations/lvis_v1_train.json',
            img_prefix='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='ObjaverseAugment',
                     ignore_passthrough=False),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=False),
                dict(
                    type='Resize',
                    img_scale=[(1333, 640), (1333, 672), (1333, 704),
                               (1333, 736), (1333, 768), (1333, 800)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ],
            data_root='../../datasets/coco')),
)


model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="GumbelCE"),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-2, override=dict(name='fc_cls')))))

work_dir='./experiments/gumbel_r50_4x4_1x/'