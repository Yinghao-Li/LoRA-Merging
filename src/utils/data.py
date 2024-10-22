import numpy as np


def lbs2array(lbs_list: list[str] | list[list[str]], lb2pos: dict[str, int]) -> np.ndarray:
    is_single = isinstance(lbs_list[0], str)
    if is_single:
        lbs_list = [lbs_list]

    arr = np.zeros((len(lbs_list), len(lb2pos)))

    for lbs, arr_row in zip(lbs_list, arr):
        for lb in lbs:
            if lb in lb2pos:
                arr_row[lb2pos[lb]] = 1

    if is_single:
        arr = arr.squeeze()

    return arr
