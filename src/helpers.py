import os

from src.templates import ZIKA_DATAFIELD_TO_KEEP

def setup_gpu(gpu=False):
    """
    """
    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def cast_to_bool(a):
    """
    """
    if a > 0:
        return 1
    else:
        return 0


def get_data_field(df):
    """
    """
    return df["data_field"] in ZIKA_DATAFIELD_TO_KEEP


def clean_zika_data(zika_cases):
    """
    """
    # Only keep the desired data fields
    zika_cases = zika_cases[zika_cases.apply(get_data_field, axis=1)].reset_index()
    return zika_cases