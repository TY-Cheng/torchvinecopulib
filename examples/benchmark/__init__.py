# %%
import os
import pickle
import platform
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import pyvinecopulib as pvc
import torch
from dotenv import load_dotenv
from torch.special import ndtr

import torchvinecopulib as tvc

# ! ===================
# ! ===================
is_test = False
# ! ===================
# ! ===================
num_threads = 10
is_cuda_avail = torch.cuda.is_available()
load_dotenv()
DIR_WORK = Path(os.getenv("DIR_WORK"))
DIR_OUT = DIR_WORK / "examples" / "benchmark" / "out"
DIR_OUT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)
SEED = 42
if is_test:
    lst_num_dim = [10, 30][::-1]
    lst_num_obs = [1000, 50000][::-1]
    lst_seed = list(range(3))
else:
    lst_num_dim = [10, 20, 30, 40, 50][::-1]
    lst_num_obs = [1000, 5000, 10000, 20000, 30000, 40000, 50000][::-1]
    lst_seed = list(range(100))
dct_time_fit = {
    "pvc": defaultdict(list),
    "tvc": defaultdict(list),
    "tvc_cuda": defaultdict(list),
}
dct_time_sample = {
    "pvc": defaultdict(list),
    "tvc": defaultdict(list),
    "tvc_cuda": defaultdict(list),
}
dct_time_pdf = {
    "pvc": defaultdict(list),
    "tvc": defaultdict(list),
    "tvc_cuda": defaultdict(list),
}


def cuda_warmup(num_obs, num_dim):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for _ in range(5):
            _ = torch.randn(num_obs, num_dim, device=device)
        torch.cuda.synchronize()
    else:
        print("CUDA is not available. Skipping warm-up.")


print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.splitlines()[0]}")
print(f"Platform:          {platform.platform()}")
print(
    f"PyTorch:           {torch.__version__} (CUDA available: {torch.cuda.is_available()}, CUDA toolkit: {torch.version.cuda})"
)
print(f"pyvinecopulib:     {pvc.__version__}")
print(f"torchvinecopulib:  {tvc.__version__}")
print(f"number of seeds:   {len(lst_seed)}")

# %%
for num_dim, num_obs in product(lst_num_dim, lst_num_obs):
    print(f"\n\n{time.strftime('%Y-%m-%d %H:%M:%S')}\nnum_dim: {num_dim}, num_obs: {num_obs}\n\n")
    # ! preprocess into copula scale (uniform marginals)
    torch.manual_seed(SEED)
    # * tensor on cpu
    R = torch.rand(num_dim, num_dim, dtype=torch.float64)
    R /= R.norm(dim=1, keepdim=True)
    R @= R.T
    U = ndtr(
        (torch.randn(num_obs, num_dim, dtype=torch.float64) @ torch.linalg.cholesky(R, upper=True))
    )
    # * tensor on cuda
    if is_cuda_avail:
        U_cuda = U.cuda()
    # * np on cpu
    U_numpy = U.numpy().astype("float64")
    # ! pvc
    pvc_ctrl = pvc.FitControlsVinecop(
        family_set=(pvc.BicopFamily.indep, pvc.BicopFamily.tll),
        nonparametric_method="quadratic",
        tree_criterion="tau",
        num_threads=num_threads,
    )
    # ! tvc
    tvc_mdl = tvc.VineCop(num_dim=num_dim, num_step_grid=64, is_cop_scale=True)
    # ! tvc_cuda
    if is_cuda_avail:
        tvc_mdl_cuda = tvc.VineCop(num_dim=num_dim, num_step_grid=64, is_cop_scale=True).cuda()

    # * fit
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} Fitting...\n")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PVC...")
    for seed in lst_seed:
        t0 = time.perf_counter()
        pvc_mdl = pvc.Vinecop.from_data(data=U_numpy, controls=pvc_ctrl)
        t1 = time.perf_counter()
        if seed > 0:
            dct_time_fit["pvc"][num_obs, num_dim].append(t1 - t0)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TVC...")
    for seed in lst_seed:
        t0 = time.perf_counter()
        tvc_mdl.fit(obs=U, mtd_bidep="kendall_tau", num_iter_max=11)
        t1 = time.perf_counter()
        if seed > 0:
            dct_time_fit["tvc"][num_obs, num_dim].append(t1 - t0)
    if is_cuda_avail:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TVC CUDA...")
        cuda_warmup(num_obs, num_dim)
        for seed in lst_seed:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tvc_mdl_cuda.fit(obs=U_cuda, mtd_bidep="kendall_tau", num_iter_max=11)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            if seed > 0:
                dct_time_fit["tvc_cuda"][num_obs, num_dim].append(t1 - t0)

    # * sample
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} Sampling...\n")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PVC...")
    for seed in lst_seed:
        t0 = time.perf_counter()
        pvc_mdl.simulate(n=num_obs, num_threads=num_threads)
        t1 = time.perf_counter()
        if seed > 0:
            dct_time_sample["pvc"][num_obs, num_dim].append(t1 - t0)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TVC...")
    for seed in lst_seed:
        t0 = time.perf_counter()
        tvc_mdl.sample(num_sample=num_obs)
        t1 = time.perf_counter()
        if seed > 0:
            dct_time_sample["tvc"][num_obs, num_dim].append(t1 - t0)
    if is_cuda_avail:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TVC CUDA...")
        cuda_warmup(num_obs, num_dim)
        for seed in lst_seed:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tvc_mdl_cuda.sample(num_sample=num_obs)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            if seed > 0:
                dct_time_sample["tvc_cuda"][num_obs, num_dim].append(t1 - t0)

    # * pdf
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} PDF...\n")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PVC...")
    for seed in lst_seed:
        t0 = time.perf_counter()
        pvc_mdl.pdf(U_numpy, num_threads=num_threads)
        t1 = time.perf_counter()
        if seed > 0:
            dct_time_pdf["pvc"][num_obs, num_dim].append(t1 - t0)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TVC...")
    for seed in lst_seed:
        t0 = time.perf_counter()
        tvc_mdl.log_pdf(U)
        t1 = time.perf_counter()
        if seed > 0:
            dct_time_pdf["tvc"][num_obs, num_dim].append(t1 - t0)
    if is_cuda_avail:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TVC CUDA...")
        cuda_warmup(num_obs, num_dim)
        for seed in lst_seed:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tvc_mdl_cuda.log_pdf(U_cuda)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            if seed > 0:
                dct_time_pdf["tvc_cuda"][num_obs, num_dim].append(t1 - t0)

# %%
# ! save
with open(DIR_OUT / "time_fit.pkl", "wb") as f:
    pickle.dump(dct_time_fit, f)
with open(DIR_OUT / "time_sample.pkl", "wb") as f:
    pickle.dump(dct_time_sample, f)
with open(DIR_OUT / "time_pdf.pkl", "wb") as f:
    pickle.dump(dct_time_pdf, f)
