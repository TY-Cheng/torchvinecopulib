# %%
import pickle
from pathlib import Path

from sklearn.metrics import d2_pinball_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from quantile_regression import DIR_OUT, SEED, get_logger
from quantile_regression.data import get_train_test_data
from quantile_regression.pipeline import get_pipe_pargrid_scorer, mean_pinball_loss

lst_data = ["diabetes", "concrete", "wine"]
lst_alpha = [0.01, 0.1, 0.5, 0.9, 0.99]
lst_model = ["rfqr", "knnqr", "lgbmqr", "vineqr"]
dct_loss = {
    "mean_pinball_loss": mean_pinball_loss,
    "d2_pinball_score": d2_pinball_score,
}
n_splits = 5
n_repeats = 3
verbose = 2
n_jobs = 10  # ! check num of physical core pls
res = {}

for data in lst_data:
    X_train, X_test, y_train, y_test = get_train_test_data(data)
    logger = get_logger(
        log_file=f"{DIR_OUT}/{data}.log",
        name=data,
    )
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"y_test: {y_test.shape}")
    for alpha in lst_alpha:
        data_alpha = f"{data}_{alpha}"
        file_out = Path(f"{DIR_OUT / data_alpha}.pkl")
        if file_out.exists():
            logger.info(f"File already exists: {file_out}")
            continue
        res_data_alpha = {}
        res_data_alpha["gs"] = {}
        res_data_alpha["mean_pinball_loss"] = {}
        res_data_alpha["d2_pinball_score"] = {}
        res_data_alpha["y_pred"] = {}
        res_data_alpha["y_true"] = y_test
        # * cross-validation
        cv = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=SEED,
        )
        for model_name in lst_model:
            logger.info(
                f"\n\n=========================\nModel: {model_name}, data: {data}, alpha: {alpha}\n========================="
            )
            pipe, pargrid, neg_pinball_loss_alpha = get_pipe_pargrid_scorer(
                alpha=alpha,
                random_state=SEED,
                model_name=model_name,
            )
            # * grid search
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=pargrid,
                scoring=neg_pinball_loss_alpha,
                cv=cv,
                verbose=verbose,
                n_jobs=n_jobs,
                refit=True,
            )
            # * fit
            gs.fit(X_train, y_train)
            if not hasattr(gs, "best_estimator_"):
                raise RuntimeError(f"{model_name} never fit successfully for α={alpha}")
            # * save results
            res_data_alpha["gs"][model_name] = gs
            res_data_alpha["y_pred"][model_name] = gs.predict(X_test)
            res_data_alpha["mean_pinball_loss"][model_name] = mean_pinball_loss(
                y_true=y_test,
                y_pred=res_data_alpha["y_pred"][model_name],
                alpha=alpha,
            )
            res_data_alpha["d2_pinball_score"][model_name] = d2_pinball_score(
                y_true=y_test,
                y_pred=res_data_alpha["y_pred"][model_name],
                alpha=alpha,
            )
            logger.info(
                f"Best params: {gs.best_params_}\n"
                f"Best score: {gs.best_score_}\n"
                f"Mean pinball loss: {res_data_alpha['mean_pinball_loss'][model_name]}\n"
                f"D2 pinball score: {res_data_alpha['d2_pinball_score'][model_name]}\n"
            )
        logger.info(
            """\n\n=========================Summary for {data_alpha}========================="""
        )
        logger.info(f"Data: {data}, alpha: {alpha}")
        for loss in dct_loss:
            logger.info(f"{loss}:")
            for model_name in lst_model:
                logger.info(f"{model_name}: {res_data_alpha[loss][model_name]}")
        with open(file_out, "wb") as f:
            pickle.dump(res_data_alpha, f)
        logger.info(f"Saved: {DIR_OUT}/{data_alpha}.pkl\n\n")
        res[data_alpha] = res_data_alpha


logger.info("All done.")
