'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:44:58
'''
import os

import yaml

import utils


class SETTINGS(object):
    # COMMAN
    EXNAME = ""
    DEVICE = "cuda: 0"
    SYNTHETIC_DEVICE = "cuda: 1"

    DEBUG = True
    RUNTIME_ROOT = "./runtime"

    PROJECT_ROOT = "."

    # DATASET
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    # DIRECT CROP
    DIRECT_CROP = False  # if use direct crop, we will render only the crop size field. which will speed up
    CROP_SIZE = [256, 192]

    @property
    def ASPECT_RATIO(self):
        return float(self.IMAGE_WIDTH) / float(self.IMAGE_HEIGHT)

    @property
    def IMAGE_SIZE(self):
        return max(self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

    MODEL_DICT = {
        "002_master_chef_can": 1,
        "003_cracker_box": 2,
        "004_sugar_box": 3,
        "005_tomato_soup_can": 4,
        "006_mustard_bottle": 5,
        "007_tuna_fish_can": 6,
        "008_pudding_box": 7,
        "009_gelatin_box": 8,
        "010_potted_meat_can": 9,
        "011_banana": 10,
        "019_pitcher_base": 11,
        "021_bleach_cleanser": 12,
        "024_bowl": 13,
        "025_mug": 14,
        "035_power_drill": 15,
        "036_wood_block": 16,
        "037_scissors": 17,
        "040_large_marker": 18,
        "051_large_clamp": 19,
        "052_extra_large_clamp": 20,
        "061_foam_brick": 21,
    }
    CLASS_ID = None

    FILELIST = {
        "train": "datasets/ycb/config/train_data_list.txt",
        "test": "datasets/ycb/config/test_data_list.txt",
    }

    IMAGE_STD = [0.229, 0.224, 0.225]
    IMAGE_MEAN = [0.485, 0.456, 0.406]

    DILATED_MASK = True
    DILATED_KERNEL_SIZE = 10
    MASK_ENLARGE = 1.4

    INIT_POSE_METHOD = "NOISE"  # NOISE, ICP, POSECNN
    # NOISE
    NOISE_ROT = 45.0 / 180.0
    NOISE_TRANS = 0.01  # m

    NUM_POINTS = 1024
    DATASET = "ycb"
    DATA_ROOT = "/media/densechen/新加卷1/ycb"
    DATA_TYPE = "OBSERVED"  # OBSERVED, SYNTHETIC

    def is_synthetic_dataset(self):
        return self.DATA_TYPE == "SYNTHETIC"

    # GROUND TRUTH
    STATE_DIM = 1024
    ROT_FORMAT = "ortho6d"  # ortho6d, euler, quat

    @property
    def ROT_DIM(self):
        return {"euler": 3, "quat": 4, "ortho6d": 6}[self.ROT_FORMAT]

    TRANS_DIM = 3

    @property
    def ACTION_DIM(self):
        return self.ROT_DIM + self.TRANS_DIM

    FX = 1066.7779541015625
    FY = 1067.487060546875
    CX = 312.9869079589844
    CY = 241.31089782714844

    def set_intrinsic(self, intrinsic: utils.Intrinsic):
        # NOTE: The intrinsic of camera at on batch must keep the same.
        self.FX = intrinsic.fx[0].item()
        self.FY = intrinsic.fy[0].item()
        self.CX = intrinsic.cx[0].item()
        self.CY = intrinsic.cy[0].item()

    # TRAIN
    BATCH_SIZE = 2
    NUM_BATCH_WHILE_SYNTHETIC = 16
    BACKBONE_LEARNING_RATE = 1e-4
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # TEST
    RESULT_DIR = "result"

    # LOSS
    MASK_LOSS = True
    MASK_LOSS_WEIGHT = 0.1

    FLOW_LOSS = True
    FLOW_LOSS_WEIGHT = 0.1

    POINT_WISE_LOSS = True
    POINT_WISE_LOSS_WEIGHT = 1.0

    EPOCH = 10
    LOG_INTERVAL = 100  # ITERATIONS
    MILESTONES = [3, 5]
    GAMMA = 0.1

    EPISODE_LEN = 1
    SYNTHETIC_EPISODE_LEN = 2
    TEST_EPISODE = 1
    MAX_EPISODE_LEN = 5

    def set_episode(self, e=1):
        self.EPISODE_LEN = e

    def set_synthetic_episode(self, e=1):
        self.SYNTHETIC_EPISODE_LEN = e

    def step_episode(self, inc=1):
        self.EPISODE_LEN = min(self.MAX_EPISODE_LEN, self.EPISODE_LEN + inc)
        return self.EPISODE_LEN

    def step_synthetic_episode(self, inc=1):
        self.SYNTHETIC_EPISODE_LEN = min(self.MAX_EPISODE_LEN,
                                         self.EPISODE_LEN + inc)
        return self.SYNTHETIC_EPISODE_LEN

    QUEUE_LEN = 100

    RL = False
    IRL = False
    VDB_UPDATE_NUM = 3
    ACTOR_CRITIC_UPDATE_NUM = 3
    IRL_GAMMA = 0.99
    IRL_LAMDA = 0.98
    CLIP_PARAM = 0.2

    @property
    def TRAINER(self):
        if self.RL:
            return "rl_trainer"
        if self.IRL:
            return "irl_trainer"
        return "trainer"

    # INPUT
    WITH_MASK = True
    WITH_IMAGE = True
    WITH_DEPTH = True

    @property
    def IN_CH(self):
        in_ch = 0
        if self.WITH_MASK:
            in_ch += 1
        if self.WITH_IMAGE:
            in_ch += 3
        if self.WITH_DEPTH:
            in_ch += 1
        return in_ch

    # RESUME
    RESUME = False
    RESUME_PATH = ""
    PRETRAIN = False
    PRETRAIN_PATH = ""

    ARCH = "flownets_bn"  # flownets, flownets_bn, flownetc, flownetc_bn
    PRETRAINED_MODEL = "pretrained/flownets_bn_EPE2.459.pth.tar"

    @property
    def SAVE_PATH(self):
        return os.path.join(self.CHECKPOINT, "{}.pth".format(self.EXNAME))

    @property
    def CHECKPOINT(self):
        return os.path.join(self.RUNTIME_ROOT, "checkpoints")

    @property
    def DEBUG_PATH(self):
        return os.path.join(self.RUNTIME_ROOT, "debug")

    def __init__(self, yaml_file=None):
        super().__init__()

        if yaml_file is not None:
            self.load_yaml(yaml_file)
        self.check_settings()

    def check_settings(self):
        # Check settings and build folders
        if self.DEBUG:
            os.makedirs(self.DEBUG_PATH, exist_ok=True)

        os.makedirs(self.CHECKPOINT, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

    def load_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            file_data = f.read()
        yaml_config = yaml.safe_load(file_data)

        for k, v in yaml_config.items():
            if k in [
                    "ASPECT_RATIO",
                    "IMAGE_SIZE",
                    "ROT_DIM",
                    "ACTION_DIM",
                    "TRAINER",
                    "IN_CH",
                    "CHECKPOINT",
                    "SAVE_PATH",
                    "DEBUG_PATH",
            ]:
                # We don't need to update the @property method.
                continue
            else:
                self.__setattr__(k, v)

    @staticmethod
    def settings_to_dict(cls_obj):
        return dict((name, getattr(cls_obj, name)) for name in dir(cls_obj)
                    if not name.startswith('__')
                    and not callable(getattr(cls_obj, name)))

    def dump_yaml(self, yaml_file):
        with open(yaml_file, "w") as f:
            yaml.safe_dump(SETTINGS.settings_to_dict(self), f)

    def merge_args(self, args):
        args_dict = SETTINGS.settings_to_dict(args)
        for k, v in args_dict.items():
            k = str(k).upper()
            if k == "DEVICE" or k == "SYNTHETIC_DEVICE":
                self.__setattr__(k, "cuda: {}".format(v) if v != "cpu" else v)
            else:
                self.__setattr__(k, v)
