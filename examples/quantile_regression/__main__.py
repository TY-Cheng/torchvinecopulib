# %%
import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

from sklearn.metrics import d2_pinball_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from quantile_regression import DIR_OUT, get_logger
from quantile_regression.data import get_train_test_data
from quantile_regression.pipeline import get_pipe_pargrid_scorer, mean_pinball_loss

is_test = False
if is_test:
    lst_data = ["diabetes"]
    lst_alpha = [0.01]
    n_repeats = 1
else:
    lst_data = ["diabetes", "wine", "housing", "concrete"]
    lst_alpha = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    n_repeats = 1000
lst_model = ["rfqr", "knnqr", "lgbmqr", "vineqr"]
dct_loss = {
    "mean_pinball_loss": mean_pinball_loss,
    "d2_pinball_score": d2_pinball_score,
}
n_splits = 5
verbose = 2
n_jobs = 10  # ! check num of physical core pls
n_iter_randomsearch = 31
for data_name in lst_data:
    X_train, X_test, y_train, y_test = get_train_test_data(data_name)
    logger = get_logger(
        log_file=f"{DIR_OUT}/{data_name}.log",
        name=data_name,
    )
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"y_test: {y_test.shape}")
    for alpha in lst_alpha:
        data_alpha = f"{data_name}_{alpha}"
        file_out = Path(f"{DIR_OUT / data_alpha}.pkl")
        if file_out.exists():
            logger.info(f"File already exists: {file_out}")
            continue
        res_data_alpha = {}
        res_data_alpha["data"] = data_name
        res_data_alpha["gs"] = defaultdict(list)
        res_data_alpha["mean_pinball_loss"] = defaultdict(list)
        res_data_alpha["d2_pinball_score"] = defaultdict(list)
        res_data_alpha["y_pred"] = defaultdict(list)
        res_data_alpha["y_true"] = y_test
        for model_name, seed in product(lst_model, range(n_repeats)):
            logger.info(
                f"\n\n\n=========================\nModel: {model_name}, data: {data_name}, alpha: {alpha}, seed: {seed}\n========================="
            )
            pipe, pargrid, neg_pinball_loss_alpha = get_pipe_pargrid_scorer(
                alpha=alpha,
                random_state=seed,
                model_name=model_name,
            )
            # * cross-validation
            cv = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            # * grid search
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=pargrid,
                scoring=neg_pinball_loss_alpha,
                n_iter=n_iter_randomsearch,
                cv=cv,
                verbose=verbose,
                n_jobs=n_jobs,
                refit=True,
                random_state=seed,
            )
            # * fit
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            if not hasattr(search, "best_estimator_"):
                raise RuntimeError(f"{model_name} never fit successfully for α={alpha}")
            # * save results
            res_data_alpha["gs"][model_name].append(search.best_params_)
            res_data_alpha["y_pred"][model_name].append(search.predict(X_test))
            res_data_alpha["mean_pinball_loss"][model_name].append(
                mean_pinball_loss(y_true=y_test, y_pred=y_pred, alpha=alpha)
            )
            res_data_alpha["d2_pinball_score"][model_name].append(
                d2_pinball_score(y_true=y_test, y_pred=y_pred, alpha=alpha)
            )
        with open(file_out, "wb") as f:
            pickle.dump(res_data_alpha, f)
        logger.info(f"Saved: {DIR_OUT}/{data_alpha}.pkl\n\n")
        logger.info(
            f"""\n\n=========================Summary for {data_alpha}========================="""
        )
        logger.info(f"Data: {data_name}, alpha: {alpha}")
        for loss in dct_loss:
            logger.info(f"{loss}:")
            for model_name in lst_model:
                logger.info(f"{model_name}: {res_data_alpha[loss][model_name]}")


logger.info("All done.")
