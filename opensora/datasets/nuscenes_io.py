import bisect
import io
import json
import os
from PIL import Image
import torch

# PYTHONPATH includes ${workspaceFolder}/externals/nuscenes-devkit/python-sdk
# or install the pip of nuscenes-devkit
from . import video_transforms
from torchvision import transforms
import random

import nuscenes.utils.splits
from nuscenes.nuscenes import NuScenes
#nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/iag/user/huangtingxuan/nuscenes', verbose=True)
from pyquaternion import Quaternion

class CanBusExtension:
    def __init__(
        self, reader, dataset_name: str = "can_bus",
        table_names: list = ["steeranglefeedback", "vehicle_monitor"]
    ):
        self.reader = reader
        self.dataset_name = dataset_name
        self.tables = {i: {} for i in table_names}
        self.indices = {i: {} for i in table_names}
        for i in reader.namelist():
            dataset, filename_ext = i.split("/")
            if dataset != "can_bus" or filename_ext == "":
                continue

            filename, ext = os.path.splitext(filename_ext)
            sp_index = filename.index("_")
            scene_name = filename[:sp_index]
            table_name = filename[sp_index + 1:]
            if table_name in table_names:
                table = json.loads(reader.read(i).decode())
                self.tables[table_name][scene_name] = table
                self.indices[table_name][scene_name] = list([
                    j["utime"] for j in table])

    def query_and_interpolate(self, table_name, scene_name, utime, column):
        if scene_name not in self.tables[table_name]:
            return None

        table = self.tables[table_name][scene_name]
        index = bisect.bisect_left(
            self.indices[table_name][scene_name], utime)
        if index == len(self.indices[table_name][scene_name]):
            return None
        elif index == 0:
            return table[0][column] if utime == table[0]["utime"] else None
        else:
            item = table[index]
            item_1 = table[index - 1]
            alpha = (utime - item_1["utime"]) / \
                (item["utime"] - item_1["utime"])
            return alpha * item[column] + (1 - alpha) * item_1[column]


class AdditionalImageAnnotation:
    def __init__(self, name, reader):
        self.name = name
        self.reader = reader
        self.keys = set(self.reader.namelist())

    def get(self, key):
        if key not in self.keys:
            return None
        else:
            return Image.open(io.BytesIO(self.reader.read(key)))


class MotionDataset(torch.utils.data.Dataset):
    table_names = [
        "calibrated_sensor", "sample", "sample_data", "scene", "sensor","sample_annotation","map", "ego_pose","log"
    ]
    index_names = [
        "calibrated_sensor.token", "sample.token",
        "sample_data.sample_token", "sample_data.token","scene.log_token",
        "scene.token", "sensor.token","sample_annotation.token","sample_annotation.sample_token",
        "map.token", "ego_pose.token","map.log_tokens","log.token",
    ]

    def get_sorted_table(tables: dict, index_name: str):
        table_name, column_name = index_name.split(".")
        sorted_table = sorted(tables[table_name], key=lambda i: i[column_name])
        index_column = [i[column_name] for i in sorted_table]
        return index_column, sorted_table

    def load_tables(
        reader, dataset_name: str, table_names: list, index_names: list
    ):
        tables = dict([
            (i, json.loads(
                reader.read("{}/{}.json".format(dataset_name, i)).decode()))
            for i in table_names])
        indices = dict([
            (i, MotionDataset.get_sorted_table(tables, i))
            for i in index_names])
        return tables, indices

    def query(
        indices: dict, table_name: str, key: str, column_name: str = "token"
    ):
        index_column, sorted_table = \
            indices["{}.{}".format(table_name, column_name)]
        i = bisect.bisect_left(index_column, key)
        return sorted_table[i]

    def query_range(
        indices: dict, table_name: str, key: str, column_name: str = "token"
    ):
        index_column, sorted_table = \
            indices["{}.{}".format(table_name, column_name)]
        i0 = bisect.bisect_left(index_column, key)
        i1 = bisect.bisect_right(index_column, key)
        return sorted_table[i0:i1] if i1 > i0 else None

    def get_scene_samples(indices, scene):
        result = []
        i = scene["first_sample_token"]
        while i != "":
            sample = MotionDataset.query(indices, "sample", i)
            result.append(sample)
            i = sample["next"]

        return result

    def is_frontal_camera(indices, sample_data):
        calibrated_sensor = MotionDataset.query(
            indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        sensor = MotionDataset.query(
            indices, "sensor", calibrated_sensor["sensor_token"])

        return sensor["modality"] == "camera" and \
            sensor["channel"] == "CAM_FRONT"

    def find_sample_data_of_nearest_time(
        sample_data_list: list, timestamp_list: list, timestamp: float
    ):
        i = bisect.bisect_left(timestamp_list, timestamp)
        t0 = timestamp - timestamp_list[i - 1]
        t1 = timestamp_list[i] - timestamp
        if i > 0 and t0 <= t1:
            i -= 1

        return sample_data_list[i]

    def enumerate_video_segments(
        sample_data_list: list, sequence_length: int, fps: float, stride: float
    ):
        timestamp_list = [i["timestamp"] for i in sample_data_list]
        sequence_duration = sequence_length / fps
        t = timestamp_list[0] / 1000000
        s = timestamp_list[-1] / 1000000 - sequence_duration
        while t <= s:
            expected_times = [
                (t + i / fps) * 1000000
                for i in range(sequence_length)
            ]
            candidates = [
                MotionDataset.find_sample_data_of_nearest_time(
                    sample_data_list, timestamp_list, i)
                for i in expected_times
            ]
            max_time_error = max([
                abs(i0["timestamp"] - i1)
                for i0, i1 in zip(candidates, expected_times)
            ])
            if max_time_error <= 500000 / fps:
                yield candidates

            t += stride

    def __init__(
        self, reader, dataset_name: str, sequence_length: int,
        fps_stride_tuples: list, split: str = None,
        image_size: tuple = None,
        full_size: tuple = None,
        training_seq_len: int = None,
        stage: str = None,
        can_bus_extension: CanBusExtension = None,
        enable_scene_description: bool = False,
        additional_image_annotations: list = None,
        temporal_sampling_scheme: dict = None,
        stub_key_data_dict: dict = {},
        metadata_dir: str = None,
        camera_list: list = None,
        **kwargs
    ):
        self.stage = stage # TTD操作加入后，必须传入此标志
        assert stage in [None, 'train', 'test'], "stage must be one of None, 'train', 'test'"
        self.training_seq_len = training_seq_len
        self.reader = reader
        tables, indices = MotionDataset.load_tables(
            self.reader, dataset_name, MotionDataset.table_names,
            MotionDataset.index_names)

        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.enable_scene_description = enable_scene_description

        self.additional_image_annotations = additional_image_annotations
        if temporal_sampling_scheme is None:
            self.temporal_sampling_scheme = {"random": 1}
        else:
            self.temporal_sampling_scheme = temporal_sampling_scheme
        # for the ego speed and steering
        self.can_bus_extension = can_bus_extension
        self.stub_key_data_dict = stub_key_data_dict
        self.image_size = image_size
        self.full_size = full_size
        if can_bus_extension is not None:
            self.stub_key_data_dict["ego_speed"] = \
                ("tensor", (sequence_length,), -1000)
            self.stub_key_data_dict["ego_steering"] = \
                ("tensor", (sequence_length,), -1000)

        self.only_samples = True
        self.load_or_generate_items(metadata_dir, dataset_name, split, tables, indices)

        def resize_nearest(img, size):
            size = (size[1], size[0])
            return img.resize(size, Image.NEAREST)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                # video_transforms.UCFCenterCropVideo(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        
        self.condition_transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_nearest(img, image_size)),
            transforms.ToTensor()
        ])
    def load_or_generate_items(self, metadata_dir, dataset_name, split, tables, indices):
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)

        file_suffix = "_items_samples.json" if self.only_samples else "_items.json"
        file_path = f"{metadata_dir}/{dataset_name}{file_suffix}"

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.items = json.load(f)
            return

        scenes = tables["scene"]
        if split is not None:
            scene_subset = getattr(nuscenes.utils.splits, split)
            scenes = [i for i in scenes if i["name"] in scene_subset]

        scene_frontal_frames = [
            (i, sorted([
                k
                for j in MotionDataset.get_scene_samples(indices, i)
                for k in MotionDataset.query_range(
                    indices, "sample_data", j["token"],
                    column_name="sample_token")
                if MotionDataset.is_frontal_camera(indices, k)
            ], key=lambda x: x["timestamp"]))
            for i in scenes
        ]

        self.items = []
        #seq len max is 192
        for i in scene_frontal_frames:
            for j in self.fps_stride_tuples:
                for k in MotionDataset.enumerate_video_segments(
                    i[1], self.sequence_length , j[0], j[1]):
                    # 检查 k 中是否有任何一个子项的字符串包含 "samples"
                    if any("samples" in item["filename"] for item in k):
                        self.items.append({"video": k, "fps": j[0], "scene": i[0]})

        
        with open(file_path, "w") as f:
            json.dump(self.items, f, indent=4)
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        if self.stage == 'train' and self.training_seq_len is not None:
            mask = [False] * len(item['video'])
            sampling_methods = list(self.temporal_sampling_scheme.keys())
            probabilities = list(self.temporal_sampling_scheme.values())
            selected_method = random.choices(sampling_methods, weights=probabilities)[0]
            if selected_method=="random":
                selected_indices = random.sample(range(len(mask)), self.training_seq_len)
            elif selected_method.startswith("stride_"):
                stride = int(selected_method.split("_")[1])
                if stride * self.training_seq_len > len(item['video']):
                    raise ValueError("The product of stride and seq_len should not exceed the video length.")
                start = random.randint(0, max(stride - 1, len(item['video']) - stride * self.training_seq_len))
                selected_indices = list(range(start, min(start + self.training_seq_len * stride, len(mask)), stride))
                # print(f"data_index={index}, stride={stride}, start={start}, selected_indices={selected_indices}")
            else:
                raise ValueError("Invalid sampling method.")
            for ind in selected_indices:
                mask[ind] = True
        else:
            mask = [True] * len(item['video'])
        pts = [
            (i["timestamp"] - item["video"][0]["timestamp"] + 500) // 1000
            for i in item["video"]
        ]
        pts = [pts[i] for i in range(len(pts)) if mask[i]]
        '''
        images = [
            Image.open(io.BytesIO(self.reader.read(i["filename"])))
            for idx, i in enumerate(item["video"]) if mask[idx]
        ]
        
        '''
        images = [
            Image.open(io.BytesIO(self.reader.read(i["filename"])))
            for idx, i in enumerate(item["video"]) 
        ]
        height = self.image_size[0]
        width = self.image_size[1]
        full_height = self.full_size[0]
        full_width = self.full_size[1]
        ar = width / height
        # ar = full_width / full_height # TODO: check ar
        result = {
            # this PIL Image item should be converted to tensor before data
            # loader collation
            "video": torch.stack([self.transforms(i) for i in images]).permute(1, 0, 2, 3).unsqueeze(0),
            #"pts": torch.tensor(pts),
            #"frame_mask": mask_tensor,
            "num_frames": self.sequence_length,
            "height": height,
            "width": width,
            "full_height": full_height,
            "full_width": full_width,
            "ar": ar,
            "fps": item["fps"],
        }
        if self.enable_scene_description:
            result["text"] = item["scene"]["description"]

        
        # extension part
        if self.can_bus_extension is not None:
            scene_name = item["scene"]["name"]
            ego_speed = [
                self.can_bus_extension.query_and_interpolate(
                    "vehicle_monitor", scene_name, i["timestamp"],
                    "vehicle_speed")
                for i in item["video"]
            ]
            if all([i is not None for i in ego_speed]):
                result["ego_speed"] = torch.tensor(ego_speed)

            ego_steering = [
                self.can_bus_extension.query_and_interpolate(
                    "steeranglefeedback", scene_name, i["timestamp"],
                    "value")
                for i in item["video"]
            ]
            if all([i is not None for i in ego_steering]):
                result["ego_steering"] = torch.tensor(ego_steering)
        if self.additional_image_annotations is not None:
            for i in self.additional_image_annotations:
                result[i.name]  = []

                for j in item["video"]:                
                    if i.name == "bbox":
                        new_filename = j["filename"]
                    else:
                        new_filename = j["filename"].replace("samples","maps/samples")
                    if i.get(new_filename) is not None:
                        result[i.name].append(self.condition_transform(i.get(new_filename)))
                    else:
                        result[i.name].append(self.condition_transform(Image.new('RGB',(100,100), 'black')))

                        
                
                result[i.name] = torch.stack(result[i.name]).unsqueeze(0)
        
        
        # add stub values for heterogeneous dataset merging
        for key, data in self.stub_key_data_dict.items():
            if key not in result.keys():
                if data[0] == "tensor":
                    shape, value = data[1:]
                    result[key] = value * torch.ones(shape)
                else:
                    result[key] = data[1]
                    
        return result
