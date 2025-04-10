import os

import av
import bisect
import json
from PIL import Image
import torch
import math
import aoss_client.client
import opensora

# The walkaround for video rotation meta (waiting for side data support on pyav
# streams)
# Note: Linux should apt install the mediainfo package for the shared library
# files.
import pymediainfo

POSSIBLE_EXTS = ["webm", "mp4", "mkv"]


class InfoExtension:
    def __init__(self, reader):
        self.reader = reader

    def query_and_interpolate(
            self, video_path: str, timestamps: list,
            table_names: list = ["locations"]
    ):
        path = video_path.replace("videos", "info/100k") \
            .replace(".mov", ".json")
        info = json.loads(self.reader.read(path).decode())

        indices = {
            i: list([j["timestamp"] for j in info[i]]) for i in table_names
        }
        result = []
        for i in timestamps:
            r = {}
            t = info["startTime"] + i
            for j in table_names:
                table = info[j]
                index = bisect.bisect_left(indices[j], t)
                if index == len(indices[j]):
                    r[j] = None
                elif index == 0:
                    r[j] = table[0] if t == table[0]["timestamp"] else None
                else:
                    item = table[index]
                    item_1 = table[index - 1]
                    alpha = (t - item_1["timestamp"]) / \
                            (item["timestamp"] - item_1["timestamp"])
                    r[j] = {
                        key: alpha * v + (1 - alpha) * item_1[key]
                        for key, v in item.items() if key != "timestamp"
                    }

            result.append(r)

        return result


class ConcatMotionDataset(torch.utils.data.Dataset):
    """
    for training recipe in vista
    """

    def __init__(self, datasets, ratios):
        self.datasets = datasets
        self.full_size = math.ceil(max([len(d) / r for d, r in zip(datasets, ratios)]))
        self.range = torch.cumsum(torch.tensor([int(r * self.full_size) for r in ratios]), dim=0)

    def __len__(self):
        return self.full_size

    def __getitem__(self, index):
        for i, r in enumerate(self.range):
            if index < r:
                cid = index % len(self.datasets[i])
                return self.datasets[i][cid]
        raise Exception(f"invalid index {index}")


class MotionDataset(torch.utils.data.Dataset):
    """
    origin open-dv dataset
    """

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

    def __init__(
            self, reader, root, client_config_path, sequence_length: int, fps_stride_tuples: list,
            info_extension: InfoExtension = None,
            ignore_list: list = ["bdd100k/videos/val/c4742900-81aa45ae.mov"],
            stub_key_data_dict: dict = {},
            meta_path='/mnt/afs/user/nijingcheng/workspace/codes/sup_codes2/data/OpenDV-YouTube.json'
    ):
        self.reader = reader
        self.root = root
        self.client = aoss_client.client.Client(client_config_path)
        self.sequence_length = sequence_length

        # for the ego speed
        self.info_extension = info_extension
        self.stub_key_data_dict = stub_key_data_dict
        if info_extension is not None:
            self.stub_key_data_dict["ego_speed"] = \
                ("tensor", (sequence_length,), -1000)

        self.items, self.metas = [], dict()
        for meta in json.load(open(meta_path, 'r')):
            self.metas[meta['videoid']] = meta

        for videoname in self.client.list('s3://' + self.root):
            videoid = videoname.split('.')[0]
            meta = self.metas[videoid]
            start_discard, end_discard = meta['start_discard'], meta['end_discard']
            length = meta['length']
            if videoid in ignore_list:
                continue
            i = os.path.join(self.root, videoid)
            for ext in POSSIBLE_EXTS:
                if self.client.contains(f"s3://{i}.{ext}"):
                    i = f'{i}.{ext}'
                    break

            f = self.reader.open(i)
            with av.open(f) as container:
                stream = container.streams.video[0]
                if stream.duration is None:
                    vid_duration = container.duration / 1e6
                else:
                    vid_duration = float(stream.duration * stream.time_base)
                vid_duration = vid_duration - start_discard - end_discard
                time_base = stream.time_base
                start_time = (stream.start_time * time_base + start_discard) / time_base

                for fps, stride in fps_stride_tuples:
                    sequence_duration = sequence_length / fps
                    t = float(start_time * time_base)
                    s = float(vid_duration - sequence_duration)
                    while t <= s:
                        self.items.append((i, t, fps))
                        t += stride

            f.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        name, t, fps = self.items[index]
        try:
            with self.reader.open(name) as f:

                # get video rotation meta
                f.seek(0)

                # decode frames
                frames = []
                with av.open(f) as container:
                    stream = container.streams.video[0]
                    time_base = stream.time_base
                    first_pts = int((t - 0.5 / fps) / time_base)
                    last_pts = int(
                        (t + (self.sequence_length + 0.5) / fps) / time_base)
                    container.seek(first_pts, stream=stream)
                    for i in container.decode(stream):
                        if i.pts < first_pts:
                            continue
                        elif i.pts > last_pts:
                            break

                        frames.append(i)

                    stream.codec_context.close()

            pts_list = [i.pts for i in frames]
        except Exception as e:
            print(f"Skip data item with index {index}")
            return self.__getitem__((index + 1) % len(self.items))

        try:
            expected_ptss = [
                int((t + i / fps) / time_base)
                for i in range(self.sequence_length)
            ]
            candidates = [
                MotionDataset.find_frame_of_nearest_time(
                    frames, pts_list, i)
                for i in expected_ptss
            ]

            # the flags to rotate back
            pil_rotations = {
                90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_90
            }
            pts = [
                int(1000 * (i.pts - candidates[0].pts) * time_base + 0.5)
                for i in candidates
            ]
            frame_rotation = 0
            images = [
                i.to_image().transpose(pil_rotations[frame_rotation])
                if frame_rotation != 0 else i.to_image()
                for i in candidates
            ]
            result = {
                # this PIL Image item should be converted to tensor before data
                # loader collation
                "images": images,
                "pts": torch.tensor(pts),
                "fps": fps,
                "scene_description": ""
            }

            # extension part
            if self.info_extension is not None:
                info_t = self.info_extension.query_and_interpolate(
                    name,
                    [
                        i + float(1000 * candidates[0].pts * time_base)
                        for i in pts
                    ])
                if all([
                    i is not None and "locations" in i and
                    i["locations"] is not None
                    for i in info_t
                ]):
                    result["ego_speed"] = torch.tensor(
                        [i["locations"]["speed"] for i in info_t])

        except Exception as e:
            print("Data item WARNING: Name {}, time {}, FPS {}, frame count {}, PTS: {}".format(
                name, t, fps, len(frames), pts_list))
            result = {
                "images": [
                    Image.new("RGB", (1280, 720), (128, 128, 128))
                    for i in range(self.sequence_length)
                ],
                "pts": torch.zeros((self.sequence_length), dtype=torch.long),
                "fps": fps,
                "scene_description": ""
            }

        # add stub values for heterogeneous dataset merging
        for key, data in self.stub_key_data_dict.items():
            if key not in result.keys():
                if data[0] == "tensor":
                    shape, value = data[1:]
                    result[key] = value * torch.ones(shape)
                else:
                    result[key] = data[1]

        return result


if __name__ == '__main__':
    import dwm.fs.s3fs

    fs = {'endpoint_url': 'http://aoss.cn-sh-01.sensecoreapi-oss.cn',
          'aws_access_key_id': 'AE939C3A07AE4E6D93908AA603B9F3A9',
          'aws_secret_access_key': 'EA3CA6A34B2747AC8ED79CB1838424E0'}
    reader = opensora.fs.s3fs.ForkableS3FileSystem(**fs)
    root = 'users/nijingcheng/datasets/opendv-youtube-data'
    client_config_path = "/mnt/afs/user/wuzehuan/aoss.conf"
    cur = MotionDataset(
        reader, root,
        client_config_path,
        8, [[10, 1]]
    )
    x = cur[0]