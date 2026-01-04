import numpy as np


def generate_random_colors(nx=32, ny=32, reproduce_paper=True):
    """Generates a random uniform RGB Image using a local RNG."""
    # Create a local RNG with a fixed seed if reproduce_paper is True
    rng = np.random.default_rng(3 if reproduce_paper else None)
    # Use the local RNG to generate random numbers
    return rng.integers(0, 256, (nx, ny, 3), dtype=np.int32)
