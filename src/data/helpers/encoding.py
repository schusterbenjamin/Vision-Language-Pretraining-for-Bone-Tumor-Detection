import torch


def encode_sex(sex: str):
    if not (sex == "F" or sex == "M"):
        raise ValueError(f"Sex must be either 'F' or 'M', got {sex}")

    return torch.Tensor([1]) if sex == "F" else torch.Tensor([0])


def encode_age(age: int):
    if age < 0:
        raise ValueError(f"Age must be a positive integer, got {age}")

    # as described in Michaels thesis the age is binned into 10 year interval with all 60 and above assigned to the same bin 7
    if age < 10:
        bin = 1
    elif age < 20:
        bin = 2
    elif age < 30:
        bin = 3
    elif age < 40:
        bin = 4
    elif age < 50:
        bin = 5
    elif age < 60:
        bin = 6
    else:
        bin = 7

    return torch.Tensor([bin])


def encode_anatomy_site(anatomy_site: str):

    anatomy_sites = [
        "shoulder",
        "arm",
        "upper arm",
        "elbow",
        "lower arm",
        "hand",
        "spine",
        "hip",
        "leg",
        "upper leg",
        "knee",
        "lower leg",
        "foot",
    ]

    # one hot encode the anatomy site
    if anatomy_site not in anatomy_sites:
        raise ValueError(f"Anatomy site must be one of {anatomy_sites}, got {anatomy_site}")
    
    return torch.Tensor([1 if site == anatomy_site else 0 for site in anatomy_sites])
