import os.path

import opensora.aoss
import opensora.common
from opensora.datasets import MotionDataset, CanBusExtension, AdditionalImageAnnotation
from torch.utils.data import ConcatDataset
from opensora.common import *

aoss_config_file_mp = {
    "foundation": '/mnt/iag/user/dingchenjing/aoss.conf',
    "devsft": 'to be filled',
    "devsftfj": '/mnt/iag/user/guoxi/aoss.conf',
    "default": "/mnt/iag/user/yangzhuoran/data/nuscenes/aoss.conf"
}

data_root_mp = {
    "foundation": 's3://users/wuzehuan/data/nuscenes/',
    "devsft": 's3://users/wuzehuan/data/nuscenes/',
    "devsftfj": 'iagfj2-guoxi:s3://drivescape-share/guoxi/data/nuscenes/',
    "default": "s3://drivescape-share/guoxi/data/nuscenes/"
}

additional_image_anno_path_mp = {
    "bbox": {
        "foundation": '/mnt/iag/user/tangweixuan/new_datasets_3dbox_v1.0/samples_v8/images/3dbox_v1.0_samples_v8.zip',
        "devsft": '/mnt/iag/user/tangweixuan/new_datasets_3dbox_v1.0/samples_v8/images/3dbox_v1.0_samples_v8.zip',
        "devsftfj": '/mnt/iag/user/tangweixuan/new_datasets_3dbox_v1.0/samples_v8/images/3dbox_v1.0_samples_v8.zip',
        "default": '/mnt/iag/user/yangzhuoran/data/nuscenes/3dbox/3dbox_v1.0_samples_v8.zip',
    },
    "hdmap": {
        "foundation": '/mnt/iag/user/tangweixuan/new_datasets_3dbox_v1.0/hdmap_v1.0.zip',
        "devsft": '/mnt/iag/user/tangweixuan/new_datasets_3dbox_v1.0/hdmap_v1.0.zip',
        "devsftfj": '/mnt/iag/user/tangweixuan/new_datasets_3dbox_v1.0/hdmap_v1.0.zip',
        "default": '/mnt/iag/user/yangzhuoran/data/nuscenes/hdmap/hdmap_v1.0.zip',
    }
}


def get_nuscenes(seq_len=8, fps_stride_tuples=None, version="mini", image_size=(512, 512), full_size=(512,512), training_seq_len=None,
                 temporal_sampling_scheme=None, stage=None, **kwargs):
    if os.path.exists('/mnt/iag/user/guoxi/devsftfj_flag'):
        running_partition = "devsftfj"
    elif os.path.exists('/mnt/iag/user/yangzhuoran/iagfj2_flag'):
        running_partition = "default"
    else:
        running_partition = 'foundation'
    path_to_aoss_config = aoss_config_file_mp[running_partition]
    data_root = data_root_mp[running_partition]
    if fps_stride_tuples is None:
        fps_stride_tuples = [(10, 0.4), (5, 0.2), (2, 0.1)]

    additional_image_annotations = [AdditionalImageAnnotation(name=item,
                              reader=opensora.common.StatelessZipFile(
                                  opensora.common.LazyFile(additional_image_anno_path_mp[item][running_partition])
                                )
                              )
                        for item in kwargs.get("additional_image_annotations", [])]

    if version=="mini":
        nuscenes_motion_dataset_cases = [
            (
                {
                    "reader": opensora.common.StatelessZipFile(
                        opensora.aoss.AossLazyFile(
                           path_to_aoss_config, data_root + "v1.0-mini.zip")
                    ),
                    "dataset_name": "v1.0-mini",
                    "sequence_length": seq_len,
                    "enable_scene_description": True,
                    "fps_stride_tuples": fps_stride_tuples,
                    "image_size": image_size,
                    "full_size": full_size,
                    "training_seq_len": training_seq_len,
                    "temporal_sampling_scheme": temporal_sampling_scheme,
                    "additional_image_annotations": additional_image_annotations,
                    "stage": stage,
                    "split": kwargs.get("split", None),
                    "camera_list": kwargs.get("camera_list", None),
                    "metadata_dir": kwargs.get("metadata_dir", None)
                },
            ),
        ]
        sub_datases = []
        for i_id, i in enumerate(nuscenes_motion_dataset_cases):
            sub_datases.append(MotionDataset(**i[0]))
        return ConcatDataset(sub_datases)
    if version=="trainval":
        zipfiles = [opensora.common.StatelessZipFile(
            opensora.aoss.AossLazyFile(path_to_aoss_config, data_root + "v1.0-trainval_meta.zip"))]
        zipfiles += [opensora.common.StatelessZipFile(
            opensora.aoss.AossLazyFile(path_to_aoss_config, data_root + f"v1.0-trainval{idx:02d}_blobs.zip"))
            for idx in range(1, 11)]
        nuscenes_motion_dataset_cases = {
            "reader": opensora.common.ChainedReaders(zipfiles),
            "dataset_name": "v1.0-trainval",
            "sequence_length": seq_len,
            "enable_scene_description": True,
            "fps_stride_tuples": fps_stride_tuples,
            "image_size": image_size,
            "full_size": full_size,
            "training_seq_len": training_seq_len,
            "temporal_sampling_scheme": temporal_sampling_scheme,
            "additional_image_annotations": additional_image_annotations,
            "stage": stage,
            "split": kwargs.get("split", None),
            "camera_list": kwargs.get("camera_list", None),
            "metadata_dir": kwargs.get("metadata_dir", None)
        }
        return MotionDataset(**nuscenes_motion_dataset_cases)
    else:
        raise ValueError("version must be one of mini or trainval")