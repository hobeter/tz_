from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np
import os
import re

OBS_FOLDER = "data/OBS_2023"
NWP_FOLDER = "data/NWP_2023"

INDICATOR = 'stn_id'
FEATURE_LIST = [
    'p', 't2', 'rh2', 'spd', 'dir', 'vis', 'rain6', 'rain24', 'tmax24', 'tmin24'
]
KEY_LIST = [INDICATOR] + FEATURE_LIST

BAD_OBS_TIME = [
    "2023020618", "2023102406", "2023020718", "2023010600", "2023030221",
    "2023061218", "2023031200", "2023040112", "2023040306", "2023021515",
    "2023021512", "2023012900", "2023022715", "2023021009"
]

TIME_FORMAT = "%Y%m%d%H"
TIME_LENGTH = timedelta(
    hours=3
)

def read_single_file(
    file_name: str
) -> Dict[str, Dict[str, float]]:
    """
    return:
        {
            indicator: {
                feature: value
                ...
            }
            ...
        }
    """
    res = {}
    keys = []
    with open(file_name, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if len(line.strip()) == 0:
                continue
            items = re.split(r"\s+", line.strip())
            if idx == 0:
                assert len( set(KEY_LIST) - set(items) ) == 0
                keys = items
            else:
                items = [
                    float(_) if keys[idx] != INDICATOR else _
                    for idx, _ in enumerate(items)
                ]
                dic = dict(zip(keys, items))
                res[dic[INDICATOR]] = {
                    k: v for k, v in dic.items()
                    if k != INDICATOR
                }
    return res


def load_obs() -> dict:
    """
    return:
        {
            date: {
                indicator: {
                    feature: value
                    ...
                }
                ...
            }
            ...
        }
    """
    print("loading OBS...")
    obs_files = [
        os.path.join(OBS_FOLDER, f, ff)
        for f in os.listdir(OBS_FOLDER) if f.startswith("OBS")
        for ff in os.listdir(os.path.join(OBS_FOLDER, f)) if ff.startswith("OBS")
    ]
    print(f"Found {len(obs_files)} OBS files")
    obs_data = {
        file_name.split("_")[-1].split(".")[0]: read_single_file(file_name)
        for file_name in obs_files
    }
    return obs_data


def load_nwp() -> dict:
    """
    return:
        {
            date: {
                hour: {
                    indicator: {
                        feature: value
                        ...
                    }
                    ...
                }
                ...
            }
            ...
        }
    """
    print("loading NWP...")
    res = {}
    nwp_dates = [
        f.split("_")[-1]
        for f in os.listdir(NWP_FOLDER) if f.startswith("NWP") 
    ]
    for nwp_date in nwp_dates:
        nwp_files = [
            os.path.join(NWP_FOLDER, f"NWP_{nwp_date}", f)
            for f in os.listdir(os.path.join(NWP_FOLDER, f"NWP_{nwp_date}"))
            if f.startswith("NWP")
        ]
        nwp_files.sort()

        tmp = {}
        for f in nwp_files:
            single_data = read_single_file(f)
            if len(single_data) == 0:
                tmp = {}
                continue
            tmp[int(f.split(".")[-1])] = single_data
        if len(tmp) != 0:
            res[nwp_date] = tmp
    return res


def add_target(
    nwp_data: dict,
    obs_data: dict,
    null_padding: float = 0.0
) -> dict:
    """
    return:
        {
            date: {
                hour: {
                    indicator: {
                        feature: value,
                        f"{feature}_target_abs": target_value,
                        f"{feature}_target_delta_pred": delta_value
                        ...
                    }
                    ...
                }
                ...
            }
            ...
        }
    """
    print("adding target...")
    for date, nwp_date in nwp_data.items():
        for hour, nwp_hour in nwp_date.items():
            if (datetime.strptime(date, TIME_FORMAT) + timedelta(hours=hour)).\
                strftime(TIME_FORMAT) in BAD_OBS_TIME:
                continue
            for indicator, nwp_indicator in nwp_hour.items():
                for feature, value in list(nwp_indicator.items()):
                    if value == 9999.00:
                        nwp_indicator[feature] = null_padding
                    nwp_indicator[f"{feature}_target_abs"] =\
                        obs_data[
                            (
                                datetime.strptime(date, TIME_FORMAT) + timedelta(hours=hour)
                             ).strftime(TIME_FORMAT)
                        ][indicator][feature]
                    nwp_indicator[f"{feature}_target_delta_pred"] =\
                        nwp_indicator[f"{feature}_target_abs"] - nwp_indicator[feature]
    return nwp_data


def gather_data(
    metadata: List[Dict[str, float]]
) -> dict:
    gathered_input = np.array([
        [
            [
                value
                for feature, value in indicator_data.items()
                if "target" not in feature
            ]
            for indicator_data in item.values()
        ]
        for item in metadata
    ])

    gathered_input_ave = np.mean(
        gathered_input.reshape(
            -1, gathered_input.shape[-1]
        ),
        axis=0
    )

    gathered_target_abs = np.array([
        [
            [
                indicator_data[f"{feature}_target_abs"]
                for feature in indicator_data.keys()
                if "target" not in feature
            ]
            for indicator_data in item.values()
        ]
        for item in metadata
    ])

    gathered_target_delta_pred = np.array([
        [
            [
                indicator_data[f"{feature}_target_delta_pred"]
                for feature in indicator_data.keys()
                if "target" not in feature
            ]
            for indicator_data in item.values()
        ]
        for item in metadata
    ])

    gathered_target_delta_ave = gathered_target_abs - gathered_input

    return {
        "gathered_input_ave": gathered_input_ave,
        "gathered_input": gathered_input,
        "gathered_target_abs": gathered_target_abs,
        "gathered_target_delta_pred": gathered_target_delta_pred,
        "gathered_target_delta_ave": gathered_target_delta_ave
    }


def split_data(
    data: dict,
    stride: int,
    overlap: int,
    indicator_list: List[str],
    feature_list: List[str]
) -> List[dict]:
    splited_data = []
    for date, date_data in data.items():
        start_idx = 0
        while (start_idx + stride) < len(date_data):
            end_idx = start_idx + stride
            
            dates = [
                (
                    datetime.strptime(date, TIME_FORMAT) + timedelta(hours=_)
                ).strftime(TIME_FORMAT)
                for _ in list(date_data.keys())[start_idx:end_idx]
            ]
            if any([
                _ in BAD_OBS_TIME
                for _ in dates
            ]):
                start_idx += stride - overlap
                continue

            metadata = list(date_data.values())[start_idx:end_idx]
            metadata = [
                {
                    indicator: {
                        feature: value
                        for feature, value in indicator_data.items()
                        if feature.split("_")[0] in feature_list or len(feature_list) == 0
                    }
                    for indicator, indicator_data in item.items()
                    if indicator in indicator_list or len(indicator_list) == 0
                } for item in metadata
            ]

            indicator_list = list(metadata[0].keys())
            feature_list = [
                _
                for _ in metadata[0][indicator_list[0]].keys()
                if "target" not in _
            ]

            gather_res = gather_data(metadata)
            splited_data.append({
                "date": date,
                "start_idx": start_idx,
                "indicator_list": indicator_list,
                "dates": dates,
                "feature_list": feature_list,
                "stride": stride,
                "overlap": overlap,
                "metadata": metadata,
                **gather_res
            })
            start_idx += stride - overlap
    return splited_data


def prepare_data(
    indicator_list: List[str] = [],
    feature_list: List[str] = [],
    eval_ratio: float = 0.1,
    stride: int = 4,
    overlap: int = 0,
    null_padding: float = 0.0
)-> dict:
    """
    args:
        indicator_list: list of indicator ids. [] for all indicators.
        feature_list: list of features. [] for all features.
        eval_ratio: ratio of date for evaluation
        stride: stride of the sliding window
        overlap: overlap of the sliding window
    """
    assert 0 <= eval_ratio <= 1
    assert stride > 0
    assert overlap >= 0
    assert overlap < stride
    
    data = add_target(
        nwp_data=load_nwp(),
        obs_data=load_obs(),
        null_padding=null_padding
    )
    eval_num = int(len(data) * eval_ratio)

    if eval_num == 0:
        train_data = data
    else:
        train_data = {
            k: v
            for idx, (k, v) in enumerate(data.items())
            if idx < len(data) - eval_num
        }
    print(f"Train date: {len(train_data)}")
    splited_train_data = split_data(
        train_data,
        stride,
        overlap,
        indicator_list,
        feature_list
    )
    print(f"Train data: {len(splited_train_data)}")

    if eval_num == 0:
        splited_eval_data = None
        print("Eval data: 0")
    else:
        eval_data = {
            k: v
            for idx, (k, v) in enumerate(data.items())
            if idx >= len(data) - eval_num
        }
        print(f"Eval date: {len(eval_data)}")
        splited_eval_data = split_data(
            eval_data,
            stride,
            overlap,
            indicator_list,
            feature_list
        )
        print(f"Eval data: {len(splited_eval_data)}")

    return {
        "indicator_list": indicator_list,
        "feature_list": feature_list,
        "eval_ratio": eval_ratio,
        "stride": stride,
        "overlap": overlap,
        "null_padding": null_padding,

        "train_data": splited_train_data,
        "eval_data": splited_eval_data
    }


def _test():
    data = prepare_data(
        indicator_list=[],
        feature_list=[],
        eval_ratio=0.1,
        stride=4,
        overlap=0,
        null_padding=0.0
    )
    print(len(data["train_data"]))
    print(data["train_data"][0].keys())
    for key, value in data["train_data"][0].items():
        print(key)
        if isinstance(value, np.ndarray):
            print(value.shape)
        elif isinstance(value, dict):
            print(value.keys())
        elif key == "metadata":
            continue
        else:
            print(value)


if __name__ == "__main__":
    _test()
