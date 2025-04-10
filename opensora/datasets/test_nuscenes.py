import unittest
import opensora.aoss
import opensora.common
from opensora.datasets import MotionDataset, CanBusExtension

class DatasetsTest(unittest.TestCase):
    def setUp(self):
        self.nuscenes_motion_dataset_cases = [
            (
                {
                    "reader": opensora.common.StatelessZipFile(
                        opensora.aoss.AossLazyFile(
                           '/mnt/iag/user/dingchenjing/aoss.conf',
                            "s3://users/wuzehuan/data/nuscenes/v1.0-mini.zip")
                    ),
                    "dataset_name": "v1.0-mini",
                    "sequence_length": 8,
                    "enable_scene_description": True,
                    "fps_stride_tuples": [(10, 0.4)]
                },
                {
                    "expected_len": 10,
                    "expected_pts_list": [
                        [0, 50, 150, 300, 400, 500, 550, 650],
                        [0, 100, 150, 250, 400, 500, 600, 650],
                        [0, 100, 150, 250, 400, 500, 600, 650],
                        [0, 50, 250, 300, 400, 500, 550, 750],
                        [0, 100, 150, 250, 400, 500, 600, 650],
                        [0, 100, 200, 250, 450, 500, 600, 700],
                        [0, 100, 250, 350, 400, 500, 600, 750],
                        [0, 100, 200, 250, 450, 500, 600, 700],
                        [0, 100, 200, 250, 450, 500, 600, 700],
                        [0, 100, 200, 250, 450, 500, 600, 700]
                    ]
                }
            ),
            (
                {
                    "reader": opensora.common.StatelessZipFile(
                        opensora.aoss.AossLazyFile(
                            '/mnt/iag/user/dingchenjing/aoss.conf',
                            "s3://users/wuzehuan/data/nuscenes/v1.0-mini.zip")
                    ),
                    "dataset_name": "v1.0-mini",
                    "sequence_length": 8,
                    "fps_stride_tuples": [(10, 30)],
                    "split": "mini_val"
                },
                {
                    "expected_len": 2,
                    "expected_pts_list": [
                        [0, 100, 150, 250, 400, 500, 600, 650],
                        [0, 100, 250, 350, 400, 500, 600, 750]
                    ]
                }
            )

        ]



    def test_nuscenes_motion_dataset(self):
        # NOTE: the test of nuScenes dataset requires network connection to the
        # AOSS
        for i_id, i in enumerate(self.nuscenes_motion_dataset_cases):
            d = MotionDataset(**i[0])
            data = [item for item in d]
            import  ipdb
            ipdb.set_trace()
            assert len(d) == i[1]["expected_len"], \
                "the length of dataset {} is wrong.".format(i_id)

            if "expected_pts_list" in i[1]:
                pts_list = [j["pts"].tolist() for j in d]
                assert all([
                    all([k[0] == k[1] for k in zip(j[0], j[1])])
                    for j in zip(pts_list, i[1]["expected_pts_list"])
                ]), "the decoded PTS of the dataset {} is different."\
                    .format(i_id)

            if "expected_ego_speed" in i[1]:
                es_list = [j["ego_speed"].tolist() for j in d]
                assert all([
                    all([abs(k[0] - k[1]) < 0.1 for k in zip(j[0], j[1])])
                    for j in zip(es_list, i[1]["expected_ego_speed"])
                ]), "the ego speed of the dataset {} is different."\
                    .format(i_id)

if __name__ == '__main__':
    nus = DatasetsTest()
    nus.setUp()
    nus.test_nuscenes_motion_dataset()
