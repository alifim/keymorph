import torchio as tio
from keymorph.utils import rescale_intensity

TRANSFORM = tio.Compose(
    [
        tio.ToCanonical(),
        # tio.Mask(masking_method="mask"),
        # tio.Resize(128),
        tio.Lambda(rescale_intensity),
        # tio.RandomNoise(mean=0, std=0.25),  # Adding Gaussian noise
        # tio.RandomBiasField(coefficients=(0, 0.5)),  # Adding MRI bias field artifact
    ]
)

EVAL_METRICS = [
    "mse",
    "softdice",
    "harddice",
    "hausd",
    "ssim",
    # "RightKidney",
    # "LeftKidney",
    # "Spleen",
    # "Liver",
    # "jdstd",
    # "jdlessthan0",
]

EVAL_UNI_NAMES = [
    # ("T1", "T1"),
    # ("T2", "T2"),
    # ("PD", "PD"),
    ("fixed", "moving"),
]
EVAL_MULTI_NAMES = [
    # ("T1", "T2"),
    # ("T1", "PD"),
    # ("T2", "PD"),
]
EVAL_LESION_NAMES = None
EVAL_GROUP_NAMES = None
EVAL_LONG_NAMES = None

EVAL_AUGS = [
    "rot0",
    # "rot45",
    # "rot90",
    # "rot135",
    # "rot180",
]

EVAL_KP_ALIGNS = [
    # "rigid",
    "affine",
    "tps_0",
    "tps_0.00001",
    "tps_0.0001",
    "tps_0.001",
    "tps_0.01",
    "tps_0.1",
    "tps_1",
    "tps_10",
    # "tps_100",
    # "tps_1000",
    # "tps_10000",
    # "tps_1e6",
    # "tps_1e7",
    # "tps_1e9",
]
