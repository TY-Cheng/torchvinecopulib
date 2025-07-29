import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()
DIR_WORK = Path(os.getenv("DIR_WORK"))
DIR_OUT = DIR_WORK / "examples" / "vcae_more" / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config_grid(path="config.yaml") -> list[SimpleNamespace]:
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    fixed_cfg = {k: v for k, v in raw_cfg.items() if k != "search"}
    grid_cfg = raw_cfg.get("search", {})

    # * product of all combinations in search
    keys, values = zip(*grid_cfg.items()) if grid_cfg else ([], [])
    configs = []
    for combo in itertools.product(*values):
        sweep_cfg = dict(zip(keys, combo))
        merged_cfg = {**fixed_cfg, **sweep_cfg}
        configs.append(SimpleNamespace(**merged_cfg))
    return configs


if __name__ == "__main__":
    # ! show number of configs
    configs = load_config_grid(DIR_WORK / "examples" / "vcae_more" / "config.yaml")
    print(f"Total configurations loaded:\t{len(configs)}")
    print("Example configuration:")
    print(vars(configs[0]))
