from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn_quantile import KNeighborsQuantileRegressor, RandomForestQuantileRegressor

from quantile_regression import SEED, VineQuantileRegressor


# %%
def get_pipe_pargrid_scorer(
    alpha: float,
    random_state: int = SEED,
    model_name: str = None,
):
    # ! scoring
    neg_pinball_loss_alpha = make_scorer(
        score_func=mean_pinball_loss,
        alpha=alpha,
        greater_is_better=False,
    )
    if model_name == "vineqr":
        # * vine
        pipe = Pipeline(
            [
                ("qr", VineQuantileRegressor(alpha=alpha)),
            ]
        )
        pargrid = {
            "qr__alpha": [alpha],
            "qr__mtd_vine": ["rvine", "cvine", "dvine"],
            "qr__mtd_bidep": [
                "kendall_tau",
                "ferreira_tail_dep_coeff",
                "chatterjee_xi",
            ],
            "qr__thresh_trunc": [1e-5, 1e-3, 1e-1],
        }
    elif model_name == "rfqr":
        # * rfqr
        pipe = Pipeline(
            [
                ("qr", RandomForestQuantileRegressor()),
            ]
        )
        pargrid = {
            "qr__q": [alpha],
            "qr__bootstrap": [False],
            "qr__n_estimators": [50, 100],
            "qr__min_samples_split": [2, 4, 6],
            "qr__max_depth": [10, None],
            "qr__max_features": ["sqrt", "log2"],
            "qr__random_state": [random_state],
        }
    elif model_name == "knnqr":
        # * knnqr
        pipe = Pipeline(
            [
                ("qr", KNeighborsQuantileRegressor()),
            ]
        )
        pargrid = {
            "qr__q": [alpha],
            "qr__n_neighbors": [5, 10, 20, 30],
            "qr__p": [1, 2],
        }
    elif model_name == "lgbmqr":
        # * lgbmqr
        pipe = Pipeline(
            [
                ("qr", LGBMRegressor()),
            ]
        )
        pargrid = {
            "qr__alpha": [alpha],
            "qr__objective": ["quantile"],
            "qr__max_depth": [31, -1],
            "qr__subsample": [0.87, 1.0],
            "qr__colsample_bytree": [0.87, 1.0],
            "qr__reg_alpha": [0.01, 0.1],
            "qr__reg_lambda": [0.01, 0.1],
            "qr__random_state": [random_state],
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return pipe, pargrid, neg_pinball_loss_alpha
