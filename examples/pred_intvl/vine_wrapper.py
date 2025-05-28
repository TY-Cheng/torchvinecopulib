import torch
import torch._dynamo as dynamo

import torchvinecopulib as tvc


def train_vine(Z_train, Y_train, seed=42, device="cpu"):
    # * build vine cop on Z and Y together
    ZY_train = torch.cat([Z_train, Y_train.view(-1, 1)], dim=1)
    # * all except last Y column, for MST in first stage
    first_tree_vertex = tuple(range(Z_train.shape[1]))
    model_vine: tvc.VineCop = tvc.VineCop(num_dim=ZY_train.shape[1], is_cop_scale=False).to(
        device=device
    )
    model_vine.fit(
        obs=ZY_train,
        first_tree_vertex=first_tree_vertex,
        mtd_bidep="ferreira_tail_dep_coeff",
        is_tll=True,
        mtd_tll="quadratic",
        seed=seed,
    )
    return model_vine


@dynamo.disable
@torch.no_grad()
def vine_pred_intvl(model: tvc.VineCop, Z_test, alpha=0.05, seed=42, device="cpu"):
    # * assuming Zy is fitted by model; y is the last column
    num_sample = Z_test.shape[0]
    idx_quantile = Z_test.shape[1]
    sample_order = (idx_quantile,)
    # * fill marginal obs (will be handled by marginal if not is_cop_scale)
    dct_v_s_obs = {(_,): Z_test[:, [_]] for _ in range(Z_test.shape[1])}
    # * fill quantile deep in the vine (assuming cop scale)
    dct_v_s_obs[model.sample_order] = torch.full((num_sample, 1), alpha / 2, device=device)
    lower = model.sample(
        num_sample=num_sample,
        sample_order=sample_order,
        dct_v_s_obs=dct_v_s_obs,
        seed=seed,
    )[:, idx_quantile]
    dct_v_s_obs[model.sample_order] = torch.full((num_sample, 1), 0.5, device=device)
    median = model.sample(
        num_sample=num_sample,
        sample_order=sample_order,
        dct_v_s_obs=dct_v_s_obs,
        seed=seed,
    )[:, idx_quantile]
    dct_v_s_obs[model.sample_order] = torch.full((num_sample, 1), 1 - alpha / 2, device=device)
    upper = model.sample(
        num_sample=num_sample,
        sample_order=sample_order,
        dct_v_s_obs=dct_v_s_obs,
        seed=seed,
    )[:, idx_quantile]
    return median.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy()
