import random

import av
import bisect
import json
from PIL import Image
import torch
from tqdm import tqdm

# The walkaround for video rotation meta (waiting for side data support on pyav
# streams)
# Note: Linux should apt install the mediainfo package for the shared library
# files.
#import decord
import os
# import EasyDict
from easydict import EasyDict as edict
import pymediainfo
import numpy as np
import cv2
import time
from torchvision import transforms
'''
{
    "cmd": 2,
    "blip": "A car is driving on a roof.",
    "folder": "val_images/KenoVelicanstveni/wrcDSfbadXw",
    "first_frame": "000000000.jpg",
    "last_frame": "000000039.jpg"
},
/mnt/iag/user/tangweixuan/datasets/OpenDV-YouTube-Language/10hz_YouTube_train_split0.json

configs/video2img.json:
{
    "video_root": "OpenDV-YouTube/videos",
    "train_img_root": "OpenDV-YouTube/full_images",
    "val_img_root": "OpenDV-YouTube/val_images",
    "meta_info": "meta/OpenDV-YouTube.json",
    "num_workers": 90,
    "frame_rate": 10,
    "exception_file": "vid2img_exceptions.txt",
    "finish_log": "vid2img_finished.txt"
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/video2img.json')
parser.add_argument('--micro', action='store_true', default=False)

args = parser.parse_args()
video_lists, meta_configs = collect_unfinished_videos(args.config, args.micro)
'''

POSSIBLE_EXTS = ["mp4", "webm", "mkv"]
IDX_WIDTH = 9

def youtuber_formatize(youtuber):
    return youtuber.replace(" ", "_")



class MotionOpenDVDataset(torch.utils.data.Dataset):
    def find_frame_of_nearest_time(frames: list, pts_list: list, pts: int):
        i = bisect.bisect_left(pts_list, pts)
        if i >= len(frames):
            return frames[-1]

        if i > 0:
            t0 = pts - pts_list[i - 1]
            t1 = pts_list[i] - pts
            if t0 <= t1:
                i -= 1

        return frames[i]

    def preprocess_dataset_json(self, json_path, regenerate=False):
        new_json_path = json_path[:-5] + '_videoid_map.json'
        if os.path.exists(new_json_path) and (not regenerate):
            try:
                with open(new_json_path, 'r') as rf:
                    new_info = json.load(rf) 
                return new_info
            except Exception as err:
               print(err)
               pass
        
        with open(json_path, 'r') as rf:
            info = json.load(rf)
        new_info = dict()
        for item in info:
            if item['videoid'] in new_info.keys():
                print(info)
            else:
                new_info[item['videoid']] = item
        #print(new_info)
        with open(new_json_path, 'w', encoding="utf8") as wf:
            json.dump(new_info, wf)
        return new_info

    def get_annos_idx(self, videoid, start_frameidx, end_frameidx, all_num_frames, video_fps, discard_begin, discard_end):
        fps = 10
        interval = video_fps / fps
        num_frames = int( fps * (all_num_frames // video_fps - discard_begin - discard_end) )
        # discard_begin, discard_end is float time
        indices = np.array([ int(discard_begin * video_fps) + int(np.round(i * interval)) for i in range(num_frames)])
        for json_data in self.items[videoid]:
            caption_start_frame_index = indices[int(json_data["first_frame"][:-4])]
            caption_end_frame_index = indices[int(json_data["last_frame"][:-4])]
            json_data["caption_start_frame_index"] = caption_start_frame_index
            json_data["caption_end_frame_index"] = caption_end_frame_index
            if start_frameidx >= caption_start_frame_index and end_frameidx <= caption_end_frame_index:
                return json_data
        return {'':''}



    def __init__(
        self, info_json_path, info_root, video_root, sequence_length: int, fps_ratio_tuples: list,
        split='train', with_text=False,expected_vae_size=(288,512)
    ):
        self.video_root = video_root
        self.sequence_length = sequence_length
        
        self.fps_ratio_tuples = fps_ratio_tuples
        temp_ = np.asarray(self.fps_ratio_tuples)
        self.fps_elements, self.fps_weights = temp_[:, 0], temp_[:, 1]
        print('fps elements:', self.fps_elements)
        print('fps weights:', self.fps_weights)
        
        with open(info_json_path, 'r') as rf:
            self.data_info = json.load(rf)
        self.data_info_map = self.preprocess_dataset_json(info_json_path, regenerate=False)
        # for the ego speed
        # for train
        self.with_text = with_text
        if with_text:
            if split=='train':
                annos = dict()
                for split_id in range(10):
                    split = json.load(open(f"{info_root}/10hz_YouTube_train_split{str(split_id)}.json", "r"))
                    for key, value in split.items():
                        if key in annos.keys():
                            annos[key].append(value)
                        else:
                            annos[key] = value
                
            elif split=='val':
                # for val
                annos = dict()
                split = json.load(open(f"{info_root}/10hz_YouTube_val.json", "r"))
                for key, value in split.items():
                    if key in annos.keys():
                        annos[key].append(value)
                    else:
                        annos[key] = value
            self.items = annos

        self.expected_vae_size = expected_vae_size
        self.transform = transforms.Compose([
        transforms.Resize(expected_vae_size),
        # transforms.CenterCrop(expected_vae_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) 
    def __len__(self):
        return len(self.data_info)#len(self.items)

    def __getitem__(self, index: int):
        for attempts in range(10):
            try:
                
                fps = random.choices(self.fps_elements, self.fps_weights, k=1)[0]

                data_info = self.data_info[index]
                videoid = data_info['videoid']
                name = os.path.join(self.video_root, videoid + '.webm')
                discard_begin, discard_end = data_info['start_discard'], data_info['end_discard']
                
                info = pymediainfo.MediaInfo.parse(name)
                #================================================
                # todo           去掉头尾
                #================================================

                frames = []

                video = cv2.VideoCapture(name)
                video_fps = video.get(cv2.CAP_PROP_FPS)
                all_num_frames: int = video.get(7) #len(video) # int frames

                #NOTE - random choice a start frame t
                #================================================
                # *               Random select a start frame t
                #================================================
                interval = video_fps / fps
                # discard_begin, discard_end is float time
                start_num_frames = int(discard_begin * video_fps)
                end_num_frames = all_num_frames - int(discard_end * video_fps)
                s = end_num_frames - np.round(self.sequence_length * interval)
                t = random.randint(start_num_frames, s)

                

                indices = np.array([ t + int(np.round(i * interval)) for i in range(self.sequence_length)])
                language_annos = {'':''}
                if self.with_text:
                    language_annos = self.get_annos_idx(videoid, indices[0], indices[-1], all_num_frames, video_fps, discard_begin, discard_end)

                for frame_index in indices:
                    if frame_index < all_num_frames:
                        

                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        _, frame = video.read()
                        frames.append(edict(pts=frame_index/video_fps, frame=Image.fromarray(frame[:, :, ::-1])))
                    else:
                        print(f"frame idx {frame_index} out of range.")

                pts_list = [i.pts for i in frames]

                
                try:

                    candidates = frames
                    # the flags to rotate back
                    pil_rotations = {
                        90: Image.Transpose.ROTATE_270,
                        180: Image.Transpose.ROTATE_180,
                        270: Image.Transpose.ROTATE_90
                    }
                    pts = [
                        int(1000 * (i.pts - candidates[0].pts) + 0.5)
                        for i in candidates
                    ]
                    rotation = info.video_tracks[0].rotation if (info.video_tracks[0].rotation) else 0
                    frame_rotation = int(float(rotation))
                    images = [
                        i.frame.transpose(pil_rotations[frame_rotation])
                        if frame_rotation != 0 else i.frame
                        for i in candidates
                    ]

                    '''
                    language_annos:
                    {
                        "cmd": 2,
                        "blip": "A car is driving on a roof.",
                        "folder": "val_images/KenoVelicanstveni/wrcDSfbadXw",
                        "first_frame": "000000000.jpg",
                        "last_frame": "000000039.jpg"
                    }
                    '''
                    result = {
                        # this PIL Image item should be converted to tensor before data
                        # loader collation
                        "video": images,
                        "pts": torch.tensor(pts),
                        "fps": fps,
                        #"language_annos":language_annos,
                        #"imgfiles": "@".join([name])
                    }
                    result["video"] = torch.stack([self.transform(i) for i in result["video"]]).permute(1, 0, 2, 3)
                    result["height"] = result["video"].shape[2]
                    result["width"] = result["video"].shape[3]
                    result["ar"] = result["width"]/result["height"] 
                    result["text"] = language_annos['blip']
                    result["num_frames"] = self.sequence_length
                    
                except Exception as e:
                    print(e)

                    print("Data item WARNING: Name {}, time {}, FPS {}, frame count {}, PTS: {} err: {}".format(
                        name, t, fps, len(frames), pts_list, repr(e)))                        
                    images= [
                            Image.new("RGB", (1280, 720), (128, 128, 128))
                            for i in range(self.sequence_length)
                        ]
                    result = {

                        "pts": torch.zeros((self.sequence_length), dtype=torch.long),
                        "fps": fps,
                        #"language_annos":language_annos
                    }
                    result["video"] = torch.stack([self.transform(i) for i in images]).permute(1, 0, 2, 3)
                    result["height"] = result["video"].shape[2]
                    result["width"] = result["video"].shape[3]
                    result["ar"] = result["width"]/result["height"] 
                    result["text"] = ""
                    result["num_frames"] = self.sequence_length
                return result
            except Exception as err:
                index = random.randint(0, len(self))
                print(err)
        raise Exception("Exceeded maximum number of attempts in opendv2k dataloader")
from torchvision.io import write_video
from torchvision.utils import save_image


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(0, 1)):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            # x.clamp_(min=low, max=high)
            # x.sub_(low).div_(max(high - low, 1e-5))
            if x.min() < 0 or x.max() > 1:
                x = (x - x.min()) / (x.max() - x.min())

        # x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        x = x.permute(1, 2, 3, 0)
        x = (x * 255).byte().to("cpu")
        write_video(save_path, x, fps=fps, video_codec="h264", options={'crf': '0'})
    print(f"Saved to {save_path}")
    


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    info_json_path = "/mnt/iag/user/tangweixuan/DriveAGI/opendv/meta/OpenDV-YouTube.json"
    info_root = "/mnt/iag/share/OpenDV2K/OpenDV-YouTube-Language_processed"
    video_root = "/mnt/iag/share/OpenDV2K/opendv-youtube-data"
    split = "train"
    with_text = True
    fps_stride_tuples = [
        [
            10,
            0.4
        ],
        [
            5,
            0.2
        ],
        [
            2,
            0.1
        ]
    ]
    sequence_length = 16
    

    class DatasetAdapter(torch.utils.data.Dataset):
        def __init__(
            self, base_dataset: torch.utils.data.Dataset):
            self.base_dataset = base_dataset
            expected_vae_size = [288,512]
            self.transform = transforms.Compose([
            transforms.Resize(expected_vae_size),
            # SmallestMaxSize(max_size),
            # transforms.CenterCrop(expected_vae_size),
            transforms.ToTensor()
        ])
            
            
        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index: int):
            item = self.base_dataset[index]
            # [x.save(f"./outputs/batch_{index}_{idx}.jpg") for idx, x in enumerate(item["images"])]
            # item["clip_images"] = torch.stack([self.clip_transform(i) for i in item["images"]])
            #print(item["images"])
            item["video"] = torch.stack([self.transform(i) for i in item["video"]])

            return item
    
    base_dataset = MotionOpenDVDataset(info_json_path, info_root, video_root, sequence_length, fps_stride_tuples, split, with_text)
    dataloader = DataLoader(base_dataset, batch_size=1, num_workers=0)

    config = {}
    config["preview_image_size"] = [576, 1024]
    from tqdm import tqdm
    for global_step, batch in tqdm(enumerate(dataloader)):
        #save_sample(batch['images'][0].permute(1,0,2,3), fps=8, save_path=f"/mnt/iag/user/zhangxin/worldsim/data/{global_step}_mode", normalize=True, value_range=(-1, 1))
        rows = len(batch['video'][0]) // 4
        preview_image = Image.new(
            "RGB",
            (4 * config["preview_image_size"][0],
             rows * config["preview_image_size"][1]), "black")
        for i_id, i in enumerate(batch['video'][0]):
            if 'blip' not in batch['language_annos'].keys():
                continue
            image = i.permute(1,2,0)
            image = (image / 2 + 0.5).clamp(0, 1)

            image = (image * 255).cpu().float().numpy().round().astype("uint8")
            
            # image_max = torch.max(image, dim=-1, keepdim=True)[0]
            # image_min = torch.min(image, dim=-1, keepdim=True)[0]
            # print(image[:5, :5], image_max[:5, :5], image_min[:5, :5])
            # image = ((image - image_min) / (image_max - image_min + 1e-8)) * 255
            # print(image[:5, :5])
            #print(image)
            '''
            image = Image.fromarray(image)
            preview_image.paste(
                image.resize(config["preview_image_size"]),
                ((i_id // rows * config["preview_image_size"][0]),
                 (i_id % rows) * config["preview_image_size"][1]))
        
            # print(f'Draw annos in {global_step}_mode.png')
            draw = ImageDraw.Draw(preview_image)
            # print(batch['language_annos'])
            draw.text((500,100), batch['language_annos']['blip'][0] + '/cmd:' + str(batch['language_annos']['cmd'][0]), fill = (255, 0 ,0))
        
            preview_image.save(os.path.join('/mnt/iag/user/zhangxin/worldsim/data', f"{global_step}_mode.png"))
            '''