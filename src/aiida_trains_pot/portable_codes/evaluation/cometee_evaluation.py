#!/usr/bin/env python

from ase.io import read
import glob
import numpy as np
from mace.calculators import MACECalculator
import torch


def calc_rmse(dft_list, dnn_list):
    """Calculate root mean square error between DFT and DNN lists
    
    Parameters
    ----------
    dft_list : list
        List of DFT quantities
    dnn_list : list
        List of DNN quantities
    
    Returns
    -------
    rmse : float
        Root mean square error between DFT and DNN quantities
    """
    
    dft = np.array(dft_list)
    dnn = np.array(dnn_list)

    rmse = np.sqrt(np.mean(np.square(dft-dnn)))
    return rmse



def deviation(data):
    shape = data.shape
    if len(shape) == 1:
        return max(data) - min(data)
    else:
        data = np.linalg.norm(data, axis=2)
        return np.mean(np.max(data, axis=0) - np.min(data, axis=0))


def main():
    potential_files = glob.glob('potential*')
    datasets = glob.glob('dataset*')


    if torch.cuda.is_available():
        calculators = [MACECalculator(potential_file, device='cuda') for potential_file in potential_files]
    else:
        calculators = [MACECalculator(potential_file, device='cpu') for potential_file in potential_files]

    evaluated_dataset = []

    for dataset in datasets:
        atoms = read(dataset, index=':', format='extxyz')
        for atm in atoms:
            evaluated_dataset.append(atm.info)
            evaluated_dataset[-1]['cell'] = np.array(atm.get_cell())
            evaluated_dataset[-1]['positions'] = np.array(atm.get_positions())
            evaluated_dataset[-1]['symbols'] = atm.get_chemical_symbols()
            try:
                evaluated_dataset[-1]['dft_forces'] = np.array(atm.get_forces())
            except:
                pass
            n_pot = 0
            energy = []
            forces = []
            stress = []

            for calculator in calculators:
                n_pot += 1
                atm.set_calculator(calculator)

                energy.append(atm.get_potential_energy())
                forces.append(np.array(atm.get_forces()))
                stress.append(np.array(atm.get_stress(voigt=False)))
                
                evaluated_dataset[-1][f'pot_{n_pot}_energy'] = atm.get_potential_energy()
                evaluated_dataset[-1][f'pot_{n_pot}_forces'] = np.array(atm.get_forces())
                evaluated_dataset[-1][f'pot_{n_pot}_stress'] = np.array(atm.get_stress(voigt=False))
                if 'dft_energy' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'energy_rmse_pot_{n_pot}'] = calc_rmse([evaluated_dataset[-1]['dft_energy']], [evaluated_dataset[-1][f'pot_{n_pot}_energy']])
                if 'dft_forces' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'forces_rmse_pot_{n_pot}'] = calc_rmse(evaluated_dataset[-1]['dft_forces'], evaluated_dataset[-1][f'pot_{n_pot}_forces'])
                if 'dft_stress' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'stress_rmse_pot_{n_pot}'] = calc_rmse(evaluated_dataset[-1]['dft_stress'], evaluated_dataset[-1][f'pot_{n_pot}_stress'])
            evaluated_dataset[-1][f'energy_deviation'] = deviation(np.array(energy))
            evaluated_dataset[-1][f'forces_deviation'] = deviation(np.array(forces))
            evaluated_dataset[-1][f'stress_deviation'] = deviation(np.array(stress))

    np.savez('evaluated_dataset.npz', evaluated_dataset = evaluated_dataset)

if __name__ == "__main__":
    main()