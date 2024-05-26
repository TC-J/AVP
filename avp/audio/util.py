from logging import warn

import numpy as np

def _pretty_time_format(time: float | int, starting_places = 12, unit_delim="", fullunitname = False):
    units: str

    if time >= 1.:
        places = starting_places

        units = "s" if not fullunitname else "seconds"

    elif time * 1e3 >= 1.:
        places = starting_places - 3

        time *= 1e3

        units = "ms" if not fullunitname else "milliseconds"

    elif time * 1e6 >= 1.:
        places = starting_places - 6

        time *= 1e6

        units = "us" if not fullunitname else "microseconds"
    
    else:
        places = starting_places - 9

        time *= 1e9

        units = "ns" if not fullunitname else "nanoseconds"
    
    return f"{time:,.{places}f}{unit_delim}{units}"


def _standardize_array(array: np.ndarray | list):
    array = np.array(array) if isinstance( array, list) else array

    if len(list(array.shape)) == 1:
        return array[:, np.newaxis]

    else:
        return array.T


def _add_channels_as_copy(array: np.ndarray | list, channels: int):
    array = _standardize_array(array)

    if array.shape[1] >= 1: warn("don't use with more than one-source channel unless it is OK to overwrite those channels' data with the first channel's data.")

    return np.repeat(array[:, 0][:, np.newaxis], channels, axis=1)