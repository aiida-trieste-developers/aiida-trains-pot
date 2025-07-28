from ase import Atoms
import numpy as np
from scipy.optimize import curve_fit

def center(atoms)-> Atoms:
    """
    Center atoms by finding the largest vacuum gap and positioning atoms
    in the center with vacuum all around them.

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
        sorted_positions = np.append(sorted_positions, sorted_positions[0] + atoms.cell[direction, direction])  # Add fictitious atom for wrap-around
        # Find all gaps including wrap-around
        gaps = []
        gap_centers = []
        
        # Gaps between consecutive atoms
        for i in range(len(sorted_positions) - 1):
            gap_size = sorted_positions[i+1] - sorted_positions[i]
            gap_center = (sorted_positions[i] + sorted_positions[i+1]) / 2
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
    """
    Enlarge vacuum between atoms that are larger than min_vacuum to target_vacuum, center atoms in the cell, and set PBC to False for the direction where the vacuum region is bigger than target_vacuum.
    
    Parameters:
    atoms: ASE Atoms object
    min_vacuum: minimum vacuum size to consider for enlarging (Angstrom)
    target_vacuum: target vacuum size after enlarging (Angstrom)

    Returns:
    atoms: ASE Atoms object with enlarged vacuum
    """

    # Center the atoms in the cell
    atoms = center(atoms.copy())
    if min_vacuum is None or target_vacuum is None:
        return atoms
    # Analyze gaps along each direction (x, y, z)
    for direction in range(3):
        if atoms.cell[direction, direction] == 0:
            continue
            
        # Find min and max positions along this direction
        min_pos = np.min(atoms.positions[:, direction])
        max_pos = np.max(atoms.positions[:, direction])
        
        # Calculate current vacuum gap size
        atom_span = max_pos - min_pos
        current_gap = atoms.cell[direction, direction] - atom_span

        # Enlarge vacuum if it meets the criteria
        if current_gap > min_vacuum:
            # Calculate how much to increase the cell
            gap_increase = target_vacuum - current_gap
            atoms.cell[direction, direction] += gap_increase
            atoms.positions[:, direction] += gap_increase / 2  # Center atoms in the new cell
            # Set PBC to False for the direction where the vacuum region is bigger then target_vacuum
            atoms.pbc[direction] = False

    return atoms



def error_calibration(dataset, thr_energy, thr_forces, thr_stress):

    def line(x, a): return a * x

    def get_rmse(dataset, key_pattern):
        return [
            np.mean([v for k, v in el.items() if k.startswith('pot_') and k.endswith(key_pattern)]) 
            for el in dataset
        ]

    
    dataset = dataset.get_list()

    RMSE_e = [e / len(el['positions']) for e, el in zip(get_rmse(dataset, '_energy_rmse'), dataset)]
    RMSE_f = get_rmse(dataset, '_forces_rmse')
    RMSE_s = get_rmse(dataset, '_stress_rmse')

    #RMSE_e = [np.mean([el['pot_4_energy_rmse'], el['pot_3_energy_rmse'], el['pot_2_energy_rmse'], el['pot_1_energy_rmse']])/len(el['positions']) for el in dataset]
    #RMSE_f = [np.mean([el['pot_4_forces_rmse'], el['pot_3_forces_rmse'], el['pot_2_forces_rmse'], el['pot_1_forces_rmse']]) for el in dataset]
    #RMSE_s = [np.mean([el['pot_4_stress_rmse'], el['pot_3_stress_rmse'], el['pot_2_stress_rmse'], el['pot_1_stress_rmse']]) for el in dataset]
    # CD_e = [el['energy_deviation_model'] for el in dataset]
    # CD_f = [el['forces_deviation_model'] for el in dataset]
    # CD_s = [el['stress_deviation_model'] for el in dataset]

    CD2_e = [el['energy_deviation'] for el in dataset]
    CD2_f = [el['forces_deviation'] for el in dataset]
    CD2_s = [el['stress_deviation'] for el in dataset]

    fit_par_e = curve_fit(line, RMSE_e, CD2_e)[0]
    fit_par_f = curve_fit(line, RMSE_f, CD2_f)[0]
    fit_par_s = curve_fit(line, RMSE_s, CD2_s)[0]

    thr_energy = fit_par_e[0] * thr_energy
    thr_forces = fit_par_f[0] * thr_forces
    thr_stress = fit_par_s[0] * thr_stress

    return thr_energy, thr_forces, thr_stress