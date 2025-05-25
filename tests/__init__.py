# conftest.py
import numpy as np
import pytest
import pyvinecopulib as pvc
import torch

import torchvinecopulib as tvc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = list(range(5))
N_SIM = 2000
EPS = tvc.util._EPS
# List the (family, true-parameter) pairs you want to test
FAMILIES = [
    (pvc.gaussian, np.array([[0.7]]), 0),  # rotation 0
    (pvc.clayton, np.array([[0.9]]), 0),
    (pvc.clayton, np.array([[0.9]]), 90),
    (pvc.clayton, np.array([[0.9]]), 180),
    (pvc.clayton, np.array([[0.9]]), 270),
    (pvc.frank, np.array([[3.0]]), 0),
    (pvc.frank, np.array([[-3.0]]), 0),
    # …add more if you like…
]


@pytest.fixture(scope="module", params=FAMILIES, ids=lambda f: f[0].name)
def bicop_pair(request):
    """
    Returns a tuple:
      ( family, true_params, U_tensor, bicop_fastkde, bicop_tll )

    notice the scope="module" so that the fixture is created only once and reused in all tests that use it.
    """
    family, true_params, rotation = request.param

    # 1) build the 'true' copula and simulate U
    true_bc = pvc.Bicop(family=family, parameters=true_params, rotation=rotation)
    U = true_bc.simulate(n=N_SIM, seeds=SEEDS)  # shape (N_SIM, 2)
    U_tensor = torch.tensor(U, device=DEVICE, dtype=torch.float64)

    # 2) fit two torchvinecopulib instances (fast KDE and TLL)
    bc_fast = tvc.BiCop(num_step_grid=512).to(DEVICE)
    bc_fast.fit(U_tensor, is_tll=False)

    bc_tll = tvc.BiCop(num_step_grid=512).to(DEVICE)
    bc_tll.fit(U_tensor, is_tll=True)

    return family, true_params, rotation, U_tensor, bc_fast, bc_tll


@pytest.fixture(scope="module")
def U_tensor():
    #  a moderately‐sized random [0,1]^2 sample
    return torch.rand(500, 2, dtype=torch.float64)


@pytest.fixture(scope="module")
def sample_1d():
    torch.manual_seed(0)
    return torch.randn(1024, 1)  # standard normal
