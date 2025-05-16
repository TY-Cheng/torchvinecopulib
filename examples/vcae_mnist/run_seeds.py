import contextlib
import logging
import sys

import pandas as pd
from tqdm import tqdm
from vcae_mnist.config import config
from vcae_mnist.experiment import run_experiment

start = int(sys.argv[1])
end = int(sys.argv[2])

# Redirect tqdm and errors to log file
log_path = f"progress_{start}_{end}.log"
log_file = open(log_path, "w")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(log_file)],
)


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        try:
            yield
        finally:
            logging.getLogger().setLevel(logging_level)


results = []

for seed in tqdm(range(start, end), desc=f"Seeds {start}-{end}", file=log_file):
    try:
        with suppress_output():
            result = run_experiment(seed, config)
        results.append(result)
    except Exception:
        logging.exception(f"Exception while running seed {seed}")

# Save results if any
try:
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"results_{start}_{end}.csv", index=False)
    logging.info(f"Saved results to results_{start}_{end}.csv")
except Exception:
    logging.exception("Failed to save results.")

log_file.close()
