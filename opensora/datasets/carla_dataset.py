
import logging

import mmcv
# from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles, Collect3D, LoadAnnotations3D

from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import torch
import os
import numpy as np
import ipdb
import json

# @DATASETS.register_module()
class CarlaDataset(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        step=16,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR", # The valid value are "LiDAR", "Camera", or "Depth".
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        start_on_keyframe=True,
        start_on_firstframe=False,
        
        
        image_size: tuple = None,
        full_size: tuple = None,
        enable_scene_description: bool = False,
        additional_image_annotations: list = None,
        annotation: dict=None,
        fps=None,
        whole_scene=False,
        dataroot=None,
        
    ) -> None:
        self.video_length = video_length
        self.start_on_keyframe = start_on_keyframe
        self.start_on_firstframe = start_on_firstframe
        self.step = step
        self.ann_file = ann_file
        self.whole_scene = whole_scene
        self.load_interval = load_interval
        self.data_infos = self.load_annotations(self.ann_file)
        
        self.root_path = dataroot
        self.image_size = image_size
        self.full_size = full_size
        self.enable_scene_description = enable_scene_description
        self.additional_image_annotations = additional_image_annotations
        self.annotation = annotation
        self.annotation['rgb'] = True
        self.fps = fps
        self.test_mode = test_mode
        
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        
        self.condition_transform = transforms.Compose([ 
            transforms.Lambda(lambda img: self.resize_nearest(img, self.image_size)),
            transforms.ToTensor(),
        ])
        

    def build_clips(self, data_infos):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_files (2-dim list of str): 2-dim list for tokens to each
            scene

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        
        all_clips = []
        if self.whole_scene:  
            clip = [token for token in data_infos]
            all_clips.append(clip)
            print("len(data_infos):", len(data_infos))
            logging.info(f"Got {len(data_infos)} "
                        f"continuous scenes. Cut into {len(clip)}-clip, "
                        f"which has {len(all_clips)} in total.")
            return all_clips

        for start in range(0, len(data_infos) - self.video_length + 1, self.step):
                
                clip = [token
                        for token in data_infos[start: start + self.video_length]]
                all_clips.append(clip)
                
        logging.info(f"Got {len(data_infos)} "
                     f"continuous scenes. Cut into {self.video_length}-clip, "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        
        data_infos = []
        with open(ann_file, 'r') as file:
            for line in file:
                data_infos.append(json.loads(line.strip()))
        
        self.clip_infos = self.build_clips(data_infos)
        
        
        return data_infos

    def __len__(self):
        return len(self.clip_infos)

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
   
        clip = self.clip_infos[index]
        # frames = []
        # for frame in clip:
        #     frame_info = super().get_data_info(frame) # 'ann_info'
        #     # info = self.data_infos[frame]
        #     frames.append(frame_info)
     
        return clip

    def resize_nearest(self, img, size):
        size = (size[1], size[0])
        return img.resize(size, Image.NEAREST)

        
    def prepare_train_data(self, index):
        """This is called by `__getitem__`
        """
        
        frames = self.get_data_info(index)
    
        if None in frames:
            return None

       
        
        examples = [] # T
        for frame in frames:
            '''
            dict_keys(['camera_infos', 'timestamp', 'ego_pose'])
            dict_keys(['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'])
            dict_keys(['extrin', 'intrin', 'image_description', 'rgb', '3dbox', 'hdmap'])
            '''
            
            
            example = {}
            
            example['rgb'] = []
            example['3dbox'] = []
            example['hdmap'] = []
            example['traj'] = []
            example['img_path'] = []
            info_list = ['rgb', '3dbox', 'hdmap']
            for cam, value in frame['camera_infos'].items(): # V    
                for key in info_list: # ['rgb', '3dbox', 'hdmap']
                    if key == 'rgb':
                        # 0000.jpg -> ' 000.jpg'
                        flag = 1
                        '''
                        directory, filename = os.path.split(frame['camera_infos'][cam][key])
                        new_name = f" {filename[1:]}"
                        new_name = os.path.join(directory, new_name)
                        frame['camera_infos'][cam][key] = new_name
                        '''
                        # ipdb.set_trace()
                    new_filename = os.path.join(self.root_path, frame['camera_infos'][cam][key])
                    
                       
                    if os.path.exists(new_filename):
                        try:
                            example[key].append(Image.open(new_filename))
                        except UnidentifiedImageError:
                            print(f"UnidentifiedImageError: cannot identify image file '{new_filename}', using black image instead.")
                            example[key].append(Image.new('RGB', (100, 100), 'black'))
                        except Exception as e:
                            print(f"Unexpected error with file '{new_filename}': {str(e)}")
                            example[key].append(Image.new('RGB', (100, 100), 'black'))
                    else:
                        print(f"File '{new_filename}' does not exist, using black image instead.")
                        example[key].append(Image.new('RGB', (100, 100), 'black'))
                
                # traj
                
                example['traj'].append(Image.new('RGB', (100, 100), 'black'))
                example['img_path'].append(new_filename)
                
               
                
            examples.append(example)    
                
        
        height = self.image_size[0]
        width = self.image_size[1]
        
 
        full_height = self.full_size[0]
        full_width = self.full_size[1]
        ar = width / height
        # ar = full_width / full_height # check ar
    
        results = {
            "video": torch.stack([torch.stack([self.transforms(i) for i in example['rgb']]) for example in examples]).permute(1, 2, 0, 3, 4), # [T, V, C, H, W] -> [V, C, T, H, W]
            "num_frames": self.video_length,
            "height": height,
            "width": width,
            "ar": ar,
            "full_height": full_height,
            "full_width": full_width,
            "hdmap": torch.stack([torch.stack([self.condition_transform(i) for i in example["hdmap"]]) for example in examples]).permute(1, 0, 2, 3, 4), # [T, V, C, H, W] -> [V, T, C, H, W]
            "bbox": torch.stack([torch.stack([self.condition_transform(i) for i in example["3dbox"]]) for example in examples]).permute(1, 0, 2, 3, 4),
            "traj": torch.stack([torch.stack([self.condition_transform(i) for i in example["traj"]]) for example in examples]).permute(1, 0, 2, 3, 4),
            "fps": self.fps,
            "img_path": [example['img_path'] for example in examples] # [T*[V]]
        }
        if self.enable_scene_description:
            results["text"] = frames[0]['camera_infos']['CAM_FRONT_LEFT']["image_description"]
        
        return results

if __name__ == "__main__":
    ann_file = "/mnt/iag/share/data/carla/data/20241112025522001_RandomRun_town05/data_infos.json"

    # cam2img
    pipeline = [
        LoadMultiViewImageFromFiles(camera_list=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]),
        LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True, with_attr_label=False), # with_bbox=True, with_label=True,),
        Collect3D(
            keys=["img", 'description', 'gt_bboxes_3d', 'gt_labels_3d'],
            meta_keys=['camera_intrinsics', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix'],
            meta_lis_keys=["filename", 'timeofday', 'location', 'token', 'description', 'cam2img']
        )
    ]
    modality = {
        "use_lidar": False,
        "use_camera": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False
    }
    dataset = CarlaDataset( ann_file,
                                step=16, # 1
                                pipeline=pipeline,
                                modality=modality,
                                start_on_firstframe=False,
                                start_on_keyframe=False,
                                video_length = 16,
                                image_size = (288, 512),
                                full_size = (288, 512 * 6),
                                enable_scene_description = True,
                                additional_image_annotations = [{'bbox': '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_test'},
                                                                {'hdmap': '/mnt/iag/user/yangzhuoran/dataset/data/hdmap_test'},
                                                                {'traj': '/mnt/iag/user/yangzhuoran/dataset/data/traj_test'},
                                                                ],
                                annotation={"hdmap":True,
                                            "bbox":True,
                                            "traj":False},
                                fps=12,
                                dataroot='/mnt/iag/share/data/carla/',
                            )
    for item in dataset:
        import pdb
        pdb.set_trace()
 
        
        
        
        