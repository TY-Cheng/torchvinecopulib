# %%
import pandas as pd
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from quantile_regression import DEVICE, DIR_DATA


def get_train_test_data(name: str, test_size=0.2, random_state=42):
    if name == "wine":
        # * wine quality dataset
        wine_quality = fetch_ucirepo(id=186)
        df_data = pd.DataFrame(
            wine_quality.data.features,
            columns=wine_quality.feature_names,
        )
    elif name == "concrete":
        # * concrete compressive strength dataset
        concrete_compressive_strength = fetch_ucirepo(id=165)
        df_data = pd.DataFrame(
            concrete_compressive_strength.data.features,
            columns=concrete_compressive_strength.feature_names,
        )
        df_data["strength"] = concrete_compressive_strength.data.targets
    elif name == "diabetes":
        df_data = pd.concat(
            [
                load_diabetes(as_frame=True)["data"],
                load_diabetes(as_frame=True)["target"],
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    df_data = df_data.dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df_data.values[:, :-1],
        df_data.values[:, -1],
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return (
        torch.from_numpy(X_train).to(DEVICE, torch.float64),
        torch.from_numpy(X_test).to(DEVICE, torch.float64),
        torch.from_numpy(y_train).to(DEVICE, torch.float64),
        torch.from_numpy(y_test).to(DEVICE, torch.float64),
    )
