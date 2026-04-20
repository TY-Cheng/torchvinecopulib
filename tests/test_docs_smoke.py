import torch

import torchvinecopulib as tvc

from . import DTYPE, gaussian_copula


def test_docs_quickstart_fit_score_and_sample():
    torch.manual_seed(0)
    obs = torch.rand(96, 4, dtype=DTYPE)

    vc = tvc.VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33)
    vc.fit(
        obs,
        mtd_vine="cvine",
        mtd_bidep="kendall_tau",
        bicop_backend="beta",
    )
    log_pdf = vc.log_pdf(obs[:8])
    samples = vc.sample(num_sample=16, seed=0)

    assert tuple(log_pdf.shape) == (8, 1)
    assert tuple(samples.shape) == (16, 4)
    assert torch.isfinite(log_pdf).all()
    assert torch.isfinite(samples).all()


def test_docs_quickstart_rosenblatt_roundtrip():
    obs = gaussian_copula(num_obs=96, rho=0.55, dim=4)

    vc = tvc.VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33)
    vc.fit(
        obs,
        mtd_vine="rvine",
        mtd_bidep="kendall_tau",
        bicop_backend="beta",
    )

    transformed = vc.rosenblatt(obs[:8])
    recovered = vc.inverse_rosenblatt(transformed)

    assert tuple(transformed.shape) == (8, 4)
    assert tuple(recovered.shape) == (8, 4)
    assert torch.isfinite(transformed).all()
    assert torch.isfinite(recovered).all()
    assert torch.allclose(recovered, obs[:8], atol=1e-6)
