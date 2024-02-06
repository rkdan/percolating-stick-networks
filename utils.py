import numpy as np
import random

def set_seed(seed=42) -> None:
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def get_seeds(N: int) -> "numpy.array":
    """Generates N random seeds that will be used to initialize the wire
        networks. N is essentially the number of simulations you'll be
        running.

    Args:
        N (int): Number of seeds

    Returns:
        numpy.array: Array of seeds
    """
    seeds = np.random.choice(np.arange(0,10000000), N, replace=False)
    return seeds


def get_time(time: float) -> str:
    """Converts time in seconds to days, hours, minutes, seconds formate

    Args:
        time (float): Time elapsed

    Returns:
        str: Reformated time
    """
    days, rem = divmod(time, 24*3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    elapsed = f"{int(days)}d, {int(hours)}hr, {int(minutes)}min, {int(seconds)}s"

    return elapsed