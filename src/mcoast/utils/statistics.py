"""
Statistical utility functions for mCOAST.

This module contains statistical utility functions including blocking analysis
and bootstrap methods for error estimation.
"""

from typing import Tuple

import numpy as np


def block_power_spectrum(
    x_data: np.ndarray, y_data: np.ndarray, growth_factor: float = 1.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple blocking for power spectrum visualization.

    Blocks paired x/y data with exponentially growing block sizes for cleaner plots.

    Args:
        x_data: Frequency data
        y_data: Power spectrum values
        growth_factor: Growth factor for block sizes

    Returns:
        Tuple of (blocked_x, blocked_y, blocked_errors)
    """
    n_data = len(x_data)

    # Calculate exponential block sizes
    n_max = int(
        np.floor(
            (np.log(1 - (n_data + 1) * (1 - growth_factor))) / np.log(growth_factor) - 1
        )
        + 1
    )
    block_sizes = [int(growth_factor**i) for i in range(1, n_max)]

    blocked_x = []
    blocked_y = []
    blocked_err = []

    n_min = 0
    for block_size in block_sizes:
        if n_min + block_size > n_data:
            break

        block_x = x_data[n_min : n_min + block_size]
        block_y = y_data[n_min : n_min + block_size]

        blocked_x.append(np.mean(block_x))
        blocked_y.append(np.mean(block_y))
        blocked_err.append(np.std(block_y) / np.sqrt(block_size))

        n_min += block_size

    return np.array(blocked_x), np.array(blocked_y), np.array(blocked_err)


def block_bispectrum(
    bispectrum: np.ndarray,
    k1_mat: np.ndarray,
    k2_mat: np.ndarray,
    num_blocks_goal: int = 75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple blocking for bispectrum visualization.

    Blocks 2D bispectrum data by averaging square regions for cleaner plots.

    Args:
        bispectrum: Bispectrum matrix
        k1_mat: k1 frequency matrix
        k2_mat: k2 frequency matrix
        num_blocks_goal: Target number of blocks

    Returns:
        Tuple of (blocked_bispectrum, blocked_k1, blocked_k2)
    """
    orig_size = bispectrum.shape[0]
    block_size = max(1, orig_size // int(np.sqrt(num_blocks_goal)))

    new_size = orig_size // block_size
    blocked_bs = np.zeros((new_size, new_size))
    blocked_k1 = np.zeros((new_size, new_size))
    blocked_k2 = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            i_start = i * block_size
            i_end = min((i + 1) * block_size, orig_size)
            j_start = j * block_size
            j_end = min((j + 1) * block_size, orig_size)

            if i_end > i_start and j_end > j_start:
                blocked_bs[i, j] = np.mean(bispectrum[i_start:i_end, j_start:j_end])
                blocked_k1[i, j] = np.mean(k1_mat[i_start:i_end, j_start:j_end])
                blocked_k2[i, j] = np.mean(k2_mat[i_start:i_end, j_start:j_end])

    return blocked_bs, blocked_k1, blocked_k2
