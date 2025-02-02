# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import sys
print(sys.path)
sys.path.append(f'{sys.path[0]}/../../')
print(sys.path)

import tempfile
from unittest.mock import MagicMock, patch

import mmcv
import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner
from torch.utils.data import DataLoader

from mmdet.core.evaluation import DistEvalHook, EvalHook
from mmdet.datasets import DATASETS, CocoDataset, CustomDataset, build_dataset
from torch.utils.data.dataloader import default_collate

from mmcv.parallel import DataContainer
from collections.abc import Mapping, Sequence
from functools import partial
import torch.nn.functional as F
from mmdet.apis import single_gpu_test


def _create_dummy_coco_json(json_name):
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
    }

    annotation_2 = {
        'id': 2,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    annotation_3 = {
        'id': 3,
        'image_id': 0,
        'category_id': 0,
        'area': 1600,
        'bbox': [150, 160, 40, 40],
        'iscrowd': 0,
    }

    annotation_4 = {
        'id': 4,
        'image_id': 0,
        'category_id': 0,
        'area': 10000,
        'bbox': [250, 260, 100, 100],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
    }]

    fake_json = {
        'images': [image],
        'annotations':
        [annotation_1, annotation_2, annotation_3, annotation_4],
        'categories': categories
    }

    mmcv.dump(fake_json, json_name)


def _create_dummy_custom_pkl(pkl_name):
    fake_pkl = [{
        'filename': 'fake_name.jpg',
        'width': 640,
        'height': 640,
        'ann': {
            'bboxes':
            np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                      [150, 160, 190, 200], [250, 260, 350, 360]]),
            'labels':
            np.array([0, 0, 0, 0])
        }
    }]
    mmcv.dump(fake_pkl, pkl_name)


def _create_dummy_results():
    boxes = [
        np.array([[50, 60, 70, 80, 1.0], [100, 120, 130, 150, 0.98],
                  [150, 160, 190, 200, 0.96], [250, 260, 350, 360, 0.95]])
    ]
    return [boxes]


def test_dataset_evaluation():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_dummy_coco_json(fake_json_file)

    # test single coco dataset evaluation
    coco_dataset = CocoDataset(
        ann_file=fake_json_file, classes=('car', ), pipeline=[])
    fake_results = _create_dummy_results()
    eval_results = coco_dataset.evaluate(fake_results, classwise=True)
    assert eval_results['bbox_mAP'] == 1
    assert eval_results['bbox_mAP_50'] == 1
    assert eval_results['bbox_mAP_75'] == 1

    # test concat dataset evaluation
    fake_concat_results = _create_dummy_results() + _create_dummy_results()

    # build concat dataset through two config dict
    coco_cfg = dict(
        type='CocoDataset',
        ann_file=fake_json_file,
        classes=('car', ),
        pipeline=[])
    concat_cfgs = [coco_cfg, coco_cfg]
    concat_dataset = build_dataset(concat_cfgs)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_bbox_mAP'] == 1
    assert eval_results['0_bbox_mAP_50'] == 1
    assert eval_results['0_bbox_mAP_75'] == 1
    assert eval_results['1_bbox_mAP'] == 1
    assert eval_results['1_bbox_mAP_50'] == 1
    assert eval_results['1_bbox_mAP_75'] == 1

    # build concat dataset through concatenated ann_file
    coco_cfg = dict(
        type='CocoDataset',
        ann_file=[fake_json_file, fake_json_file],
        classes=('car', ),
        pipeline=[])
    concat_dataset = build_dataset(coco_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_bbox_mAP'] == 1
    assert eval_results['0_bbox_mAP_50'] == 1
    assert eval_results['0_bbox_mAP_75'] == 1
    assert eval_results['1_bbox_mAP'] == 1
    assert eval_results['1_bbox_mAP_50'] == 1
    assert eval_results['1_bbox_mAP_75'] == 1

    # create dummy data
    fake_pkl_file = osp.join(tmp_dir.name, 'fake_data.pkl')
    _create_dummy_custom_pkl(fake_pkl_file)

    # test single custom dataset evaluation
    custom_dataset = CustomDataset(
        ann_file=fake_pkl_file, classes=('car', ), pipeline=[])
    fake_results = _create_dummy_results()
    eval_results = custom_dataset.evaluate(fake_results)
    assert eval_results['mAP'] == 1

    # test concat dataset evaluation
    fake_concat_results = _create_dummy_results() + _create_dummy_results()

    # build concat dataset through two config dict
    custom_cfg = dict(
        type='CustomDataset',
        ann_file=fake_pkl_file,
        classes=('car', ),
        pipeline=[])
    concat_cfgs = [custom_cfg, custom_cfg]
    concat_dataset = build_dataset(concat_cfgs)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_mAP'] == 1
    assert eval_results['1_mAP'] == 1

    # build concat dataset through concatenated ann_file
    concat_cfg = dict(
        type='CustomDataset',
        ann_file=[fake_pkl_file, fake_pkl_file],
        classes=('car', ),
        pipeline=[])
    concat_dataset = build_dataset(concat_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_mAP'] == 1
    assert eval_results['1_mAP'] == 1

    # build concat dataset through explicit type
    concat_cfg = dict(
        type='ConcatDataset',
        datasets=[custom_cfg, custom_cfg],
        separate_eval=False)
    concat_dataset = build_dataset(concat_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results, metric='mAP')
    assert eval_results['mAP'] == 1
    assert len(concat_dataset.datasets[0].data_infos) == \
        len(concat_dataset.datasets[1].data_infos)
    assert len(concat_dataset.datasets[0].data_infos) == 1
    tmp_dir.cleanup()


# @patch('mmdet.apis.single_gpu_test', MagicMock)
# @patch('mmdet.apis.multi_gpu_test', MagicMock)
# @pytest.mark.parametrize('EvalHookParam', (EvalHook, DistEvalHook))
def test_evaluation_hook(EvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, interval=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()

    evalhook = EvalHookParam(dataloader, start=1, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0/negative, interval=1: perform evaluation after each epoch and
    #    before epoch 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=0)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. start=0, interval=2, dynamic_intervals=[(3, 1)]: the evaluation
    # interval is 2 when it is less than 3 epoch, otherwise it is 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=0, interval=2, dynamic_intervals=[(3, 1)])
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 4)
    assert evalhook.evaluate.call_count == 3

    # the evaluation start epoch cannot be less than 0
    runner = _build_demo_runner()
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, start=-2)

    evalhook = EvalHookParam(dataloader, start=0)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3


def _build_demo_runner():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=logging.getLogger())
    return runner


# @pytest.mark.parametrize('classes, expected_length', [(['bus'], 2),
#                                                       (['car'], 1),
#                                                       (['bus', 'car'], 2)])
def test_allow_empty_images(classes, expected_length):
    dataset_class = DATASETS.get('CocoDataset')
    # Filter empty images
    filtered_dataset = dataset_class(
        ann_file='tests/data/coco_sample.json',
        img_prefix='tests/data',
        pipeline=[],
        classes=classes,
        filter_empty_gt=True)

    # Get all
    full_dataset = dataset_class(
        ann_file='tests/data/coco_sample.json',
        img_prefix='tests/data',
        pipeline=[],
        classes=classes,
        filter_empty_gt=False)

    assert len(filtered_dataset) == expected_length
    assert len(filtered_dataset.img_ids) == expected_length
    assert len(full_dataset) == 3
    assert len(full_dataset.img_ids) == 3
    assert filtered_dataset.CLASSES == classes
    assert full_dataset.CLASSES == classes

def test_dataset_init(data_config):
    if 'data' not in data_config:
        return
    # stage_names = ['train', 'val', 'test']
    stage_names = ['test']
    for stage_name in stage_names:
        dataset_config = copy.deepcopy(data_config.data.get(stage_name))
        dataset = build_dataset(dataset_config)
    return dataset


def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def test_model():
    config_path = '/Users/kyanchen/Code/DetFramework/configs/my_configs/fcos_NMSNet_config.py'
    config = mmcv.Config.fromfile(config_path)
    # 需修改
    # config.data_root = '/Users/kyanchen/Code/DetFramework/data/nms_net/'


    dataset = test_dataset_init(config)

    from mmdet.models import build_detector

    model_config = copy.deepcopy(config.get('model'))
    detector = build_detector(model_config)

    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=0, 
        collate_fn=partial(collate, samples_per_gpu=1)
        )
    
    outputs = single_gpu_test(detector, dataloader, with_nms=True)

    eval_kwargs = config.get('evaluation', {}).copy()
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)

if __name__ == '__main__':
    test_model()
