import torchio as tio
from keymorph.utils import rescale_intensity

# 1. The baseline validation transform (never changes)
VAL_TRANSFORM = tio.Compose([
    tio.ToCanonical(),
    tio.Lambda(rescale_intensity),
])

# 2. Define your augmentation recipes
AUGMENTATION_STRATEGIES = {
    "baseline": VAL_TRANSFORM, # Just the base processing
    
    "spatial": tio.Compose([
        tio.ToCanonical(),
        tio.Lambda(rescale_intensity),
        tio.RandomAffine(scales=0.0, degrees=15, translation=15, isotropic=True, p=0.8),
    ]),
    
    "spatial+intensity": tio.Compose([
        tio.ToCanonical(),
        tio.Lambda(rescale_intensity),
        tio.RandomAffine(scales=0.0, degrees=15, translation=15, isotropic=True, p=0.8),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.7),
        tio.RandomBiasField(coefficients=0.5, order=3, p=0.5)
    ])
}

# 3. Create a simple retrieval function
def get_train_transform(strategy_name):
    if strategy_name not in AUGMENTATION_STRATEGIES:
        raise ValueError(f"Unknown augmentation strategy: '{strategy_name}'. "
                         f"Available strategies: {list(AUGMENTATION_STRATEGIES.keys())}")
    return AUGMENTATION_STRATEGIES[strategy_name]

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
    # "tps_0.00001",
    # "tps_0.0001",
    # "tps_0.001",
    "tps_0.01",
    "tps_0.1",
    "tps_1",
    "tps_10",
    "tps_100",
    # "tps_1000",
    # "tps_10000",
    # "tps_1e6",
    # "tps_1e7",
    # "tps_1e9",
]
