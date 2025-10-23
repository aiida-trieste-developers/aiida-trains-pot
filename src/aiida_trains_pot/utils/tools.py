"""Utility functions for manipulating ASE Atoms objects and calibrating errors."""

import numpy as np

from ase import Atoms
from scipy.optimize import curve_fit


def center(atoms) -> Atoms:
    """Center atoms by finding the largest vacuum gap.

    Then position atoms in the center with vacuum all around them.

    Parameters:
    atoms: ASE Atoms object
    Returns:
    atoms: ASE Atoms object with centered positions
    """
    for direction in range(3):
        if not atoms.pbc[direction] or atoms.cell[direction, direction] == 0:
            continue

        # Sort atoms by position along this direction
        sorted_indices = np.argsort(atoms.positions[:, direction])
        sorted_positions = atoms.positions[sorted_indices, direction]
        sorted_positions = np.append(
            sorted_positions, sorted_positions[0] + atoms.cell[direction, direction]
        )  # Add fictitious atom for wrap-around
        # Find all gaps including wrap-around
        gaps = []
        gap_centers = []

        # Gaps between consecutive atoms
        for i in range(len(sorted_positions) - 1):
            gap_size = sorted_positions[i + 1] - sorted_positions[i]
            gap_center = (sorted_positions[i] + sorted_positions[i + 1]) / 2
            gaps.append(gap_size)
            gap_centers.append(gap_center)

        # Find the largest gap and move it to origin
        if gaps:
            largest_gap_idx = np.argmax(gaps)
            largest_gap_center = gap_centers[largest_gap_idx]
            atoms.positions[:, direction] -= largest_gap_center

    atoms.wrap()  # Ensure atoms are wrapped within the cell
    return atoms


def enlarge_vacuum(atoms, min_vacuum=4.0, target_vacuum=15.0) -> Atoms:
    """Enlarge vacuum between atoms that are larger than min_vacuum to target_vacuum.

    Center atoms in the cell, and set PBC to False for the direction where the vacuum
    region is bigger than target_vacuum.

    Parameters:
    atoms: ASE Atoms object
    min_vacuum: minimum vacuum size to consider for enlarging (Angstrom)
    target_vacuum: target vacuum size after enlarging (Angstrom)

    Returns:
    atoms: ASE Atoms object with enlarged vacuum
    """
    atoms = atoms.copy()

    # Convert AiiDA Float objects to Python float if needed
    if hasattr(min_vacuum, "value"):
        min_vacuum = float(min_vacuum.value)
    else:
        min_vacuum = float(min_vacuum)

    if hasattr(target_vacuum, "value"):
        target_vacuum = float(target_vacuum.value)
    else:
        target_vacuum = float(target_vacuum)

    # Ensure positions and cell are numpy arrays with proper dtype
    positions = np.array(atoms.positions, dtype=np.float64)
    cell = np.array(atoms.cell, dtype=np.float64)

    # Analyze gaps along each direction (x, y, z)
    for direction in range(3):
        if cell[direction, direction] == 0:
            continue

        # Find min and max positions along this direction
        min_pos = float(np.min(positions[:, direction]))
        max_pos = float(np.max(positions[:, direction]))

        # Calculate current vacuum gap size
        atom_span = max_pos - min_pos
        current_gap = float(cell[direction, direction]) - atom_span

        # Enlarge gap if it meets the criteria
        if current_gap > min_vacuum:
            # Calculate how much to increase the cell
            gap_increase = target_vacuum - current_gap
            cell[direction, direction] = float(cell[direction, direction]) + gap_increase
            positions[:, direction] = positions[:, direction] + (gap_increase / 2.0)
            # Set PBC to False for the direction where the vacuum region is bigger than target_vacuum
            atoms.pbc[direction] = False

    # Update atoms object with corrected arrays
    atoms.positions = positions.astype(np.float64)
    atoms.cell = cell.astype(np.float64)

    return atoms


def error_calibration(dataset, thr_energy=0.1, thr_forces=0.1, thr_stress=0.1):
    """Calibrate error thresholds based on dataset."""

    def line(x, a):
        return a * x

    def get_rmse(dataset, key_pattern):
        return [
            np.mean([v for k, v in el.items() if k.startswith("pot_") and k.endswith(key_pattern)]) for el in dataset
        ]

    def filter_nan(rmse, cd):
        rmse_arr = np.array(rmse)
        cd_arr = np.array(cd)
        mask = ~(np.isnan(rmse_arr) | np.isnan(cd_arr))
        return rmse_arr[mask].tolist(), cd_arr[mask].tolist()

    dataset = dataset.get_list()

    RMSE_e = [e / len(el["positions"]) for e, el in zip(get_rmse(dataset, "_energy_rmse"), dataset, strict=False)]
    RMSE_f = get_rmse(dataset, "_forces_rmse")
    RMSE_s = get_rmse(dataset, "_stress_rmse")

    CD2_e = [el["energy_deviation"] for el in dataset]
    CD2_f = [el["forces_deviation"] for el in dataset]
    CD2_s = [el["stress_deviation"] for el in dataset]

    RMSE_e, CD2_e = filter_nan(RMSE_e, CD2_e)
    RMSE_f, CD2_f = filter_nan(RMSE_f, CD2_f)
    RMSE_s, CD2_s = filter_nan(RMSE_s, CD2_s)

    fit_par_e = curve_fit(line, RMSE_e, CD2_e)[0]
    fit_par_f = curve_fit(line, RMSE_f, CD2_f)[0]
    fit_par_s = curve_fit(line, RMSE_s, CD2_s)[0]

    thr_energy = fit_par_e[0] * thr_energy
    thr_forces = fit_par_f[0] * thr_forces
    thr_stress = fit_par_s[0] * thr_stress

    return thr_energy, thr_forces, thr_stress
