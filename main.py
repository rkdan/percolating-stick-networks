import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from percolating_network import WireNetwork
import utils
from config.config import logger
from pathlib import Path
import os
from datetime import datetime

import typer
app = typer.Typer()


def job(size: int, seed: int) -> int:
    """Function to parallelize running multiple seeds

    Args:
        size (int): The size of the network
        seed (int): Random seed

    Returns:
        int: Critical number of wires to cause a spanning path from one side of
            the network to the other.
    """
    net = WireNetwork(size, size, seed=seed)

    return net.percolate()


def save_results(results, RES_DIR):
    seed_len = results.shape[1]
    size_len = results.shape[0]
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"{str(RES_DIR)}/Nseeds-{seed_len}_sizes-{size_len}_{dt_str}.npy", results)


@app.command()
def run(
    N_seeds: int = 8,
    sizes: str = "8,16,32,64"
) -> None:
    # Get an array of seeds
    logger.info(
        f"Total instances: {N_seeds}\nNetwork sizes: {sizes}"
    )
    # Make save path and directory
    BASE_DIR = Path(__file__).parent.parent.absolute()
    RES_DIR = Path(BASE_DIR, "Results")
    if not os.path.exists(RES_DIR):
        RES_DIR.mkdir(parents=True)

    sizes = [int(i) for i in sizes.split(",")]
    seeds = utils.get_seeds(N_seeds)
    # Create empty results array
    results = np.zeros((len(sizes), len(seeds)))

    start = time.time()
    # Run jobs
    for j, size in enumerate(tqdm(sizes, total=len(sizes), ascii="░▒█", colour='GREEN', desc="Running...")):
        # TODO: Write in a failsafe to periodically dump results to savefile.
        results [j,:] = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(job)(size, seed) for seed in seeds)
    stop = time.time()

    logger.info(f"✅ Simulations completed!\n⏱️  Time elapsed: {utils.get_time(stop-start)}")
    save_results(results, RES_DIR)


if __name__ == "__main__":
    app()
