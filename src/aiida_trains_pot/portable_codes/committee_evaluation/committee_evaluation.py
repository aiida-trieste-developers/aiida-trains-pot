#!/usr/bin/env python

from ase.io import read
import glob
import numpy as np
from mace.calculators import MACECalculator
from prettytable import PrettyTable
import torch
import re
import logging
import sys
import time

def rmse_table(RMSE) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ['SET', 'POTENTIAL', 'RMSE Energy (meV/atom)', 'RMSE Forces (meV/Å)', 'RMSE Stress (meV/Å^3/atom)']

    for key in RMSE:
        keys2 = list(RMSE[key].keys())
        if 'committee' in keys2:
            keys2.remove('committee')
            keys2 = ['committee'] + keys2
        for key2 in keys2:
            if key2 == 'committee':
                table.add_row(
                        [
                            key,
                            key2,
                            f"{RMSE[key][key2]['rmse_e'] * 1000:8.1f} ± {RMSE[key][key2]['std_e'] * 1000:<8.1f}",
                            f"{RMSE[key][key2]['rmse_f'] * 1000:8.1f} ± {RMSE[key][key2]['std_f'] * 1000:<8.1f}",
                            f"{RMSE[key][key2]['rmse_s'] * 1000:8.2f} ± {RMSE[key][key2]['std_s'] * 1000:<8.2f}",
                        ]
                    )
            else:
                table.add_row(
                    [
                        '',
                        key2.split('_')[-1],
                        f"{RMSE[key][key2]['rmse_e'] * 1000:8.1f}",
                        f"{RMSE[key][key2]['rmse_f'] * 1000:8.1f}",
                        f"{RMSE[key][key2]['rmse_s'] * 1000:8.2f}",
                    ]
                )
    return table



def global_rmse(dataset):
    """Calculate global root mean square error between DFT and DNN
    quantities and return mean and standard deviation among the
    committee of potentials.

    Parameters
    ----------
    dataset : list
        List of dictionaries containing the information of the
        evaluated dataset.

    Returns
    -------
    dict
        Dictionary containing the mean and standard deviation of
        the root mean square error for energy, forces and stress.
    """

    dnn_f = []
    dnn_e = []
    dnn_s = []
    dft_e = []
    dft_f = []
    dft_s = []
    for key, el in dataset[-1].items():
        if re.fullmatch(r'pot_\d+_forces',key):
            dnn_f.append([])
            for frame in dataset:
                dnn_f[-1].extend(frame[key].ravel())
        elif re.fullmatch(r'pot_\d+_energy',key):
            dnn_e.append([])
            for frame in dataset:
                dnn_e[-1].append(frame[key]/len(frame['positions']))
        elif re.fullmatch(r'pot_\d+_stress',key):
            dnn_s.append([])
            for frame in dataset:
                dnn_s[-1].extend(frame[key].ravel()/len(frame['positions']))
        elif 'dft_energy' in key:
            for frame in dataset:
                dft_e.append(frame[key]/len(frame['positions']))
        elif 'dft_forces' in key:
            for frame in dataset:
                dft_f.extend(frame[key].ravel())
        elif 'dft_stress' in key:
            for frame in dataset:
                dft_s.extend(frame[key].ravel()/len(frame['positions']))
    
    rmse_e = np.array([calc_rmse(dft_e, dnn) for dnn in dnn_e])
    rmse_f = np.array([calc_rmse(dft_f, dnn) for dnn in dnn_f])
    rmse_s = np.array([calc_rmse(dft_s, dnn) for dnn in dnn_s])

    RMSE = {}
    for ii, _ in enumerate(dnn_e):
        RMSE[f'pot_{ii+1}'] = {
            'rmse_e': rmse_e[ii],
            'rmse_f': rmse_f[ii],
            'rmse_s': rmse_s[ii],
        }
    if len(dnn_e) > 1:
        RMSE['committee'] = {
            'rmse_e': np.mean(rmse_e),
            'std_e': np.std(rmse_e),
            'rmse_f': np.mean(rmse_f),
            'std_f': np.std(rmse_f),
            'rmse_s': np.mean(rmse_s),
            'std_s': np.std(rmse_s),
        }

    return RMSE


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


def model_deviation(data):
    shape = data.shape
    if len(shape) == 1:
        return np.std(data)
    else:
        mean = np.mean(data, axis=0)
        return np.max(np.mean(np.linalg.norm(data - mean, axis=2)**2, axis=0)**0.5)


def maximum_deviation(data):
    shape = data.shape
    if len(shape) == 1:
        return max(data) - min(data)
    else:
        data = np.linalg.norm(data, axis=2)
        return np.mean(np.max(data, axis=0) - np.min(data, axis=0))


def main(log_freq=100):

    logging.info('###########################################')
    logging.info('### Committee evaluation of MACE models ###')
    logging.info('###########################################\n')


    potential_files = glob.glob('potential*')
    datasets = glob.glob('dataset*')

    logging.info('Loading potentials...')
    if torch.cuda.is_available():
        calculators = [MACECalculator(potential_file, device='cuda') for potential_file in potential_files]
    else:
        calculators = [MACECalculator(potential_file, device='cpu') for potential_file in potential_files]
    
    logging.info(f'Loaded {len(calculators)} potentials.\n')

   

    for jj, dataset in enumerate(datasets):
        dataset_name = dataset.replace('.xyz','')
        logging.info(f'Loading dataset {jj+1}/{len(datasets)}...')
        atoms = read(dataset, index=':', format='extxyz')
        logging.info(f'Loaded {len(atoms)} frames from dataset {jj+1}.\n')
        logging.info(f'Evaluating dataset {jj+1}/{len(datasets)}...')
        time_i = time.time()
        evaluated_dataset = []
        for ii, atm in enumerate(atoms):
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
                evaluated_dataset[-1][f'pot_{n_pot}_stress'] = np.array(atm.get_stress(voigt=True))
                if 'dft_energy' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'pot_{n_pot}_energy_rmse'] = calc_rmse([evaluated_dataset[-1]['dft_energy']], [evaluated_dataset[-1][f'pot_{n_pot}_energy']])
                if 'dft_forces' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'pot_{n_pot}_forces_rmse'] = calc_rmse(evaluated_dataset[-1]['dft_forces'], evaluated_dataset[-1][f'pot_{n_pot}_forces'])
                if 'dft_stress' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'pot_{n_pot}_stress_rmse'] = calc_rmse(evaluated_dataset[-1]['dft_stress'], np.array(atm.get_stress(voigt=True)))
            evaluated_dataset[-1][f'energy_deviation'] = maximum_deviation(np.array(energy))
            evaluated_dataset[-1][f'forces_deviation'] = maximum_deviation(np.array(forces))
            evaluated_dataset[-1][f'stress_deviation'] = maximum_deviation(np.array(stress))
            evaluated_dataset[-1][f'energy_deviation_model'] = model_deviation(np.array(energy))
            evaluated_dataset[-1][f'forces_deviation_model'] = model_deviation(np.array(forces))
            evaluated_dataset[-1][f'stress_deviation_model'] = model_deviation(np.array(stress))
            if (ii+1) % log_freq == 0:
                time_f = time.time()
                logging.info(f'Frames {ii+1:5d}/{len(atoms)} evaluated - time remaining for dataset {jj+1}: {((time_f-time_i)/(ii+1))*(len(atoms)-ii):.2f} s')

        logging.info(f'Evaluation finished for dataset {jj+1}.')
        logging.info(f'Saving evaluated dataset {jj+1}...')
        np.savez(f'{dataset_name}_evaluated.npz', evaluated_dataset = evaluated_dataset)
        logging.info(f'Saved evaluated dataset {jj+1} as {dataset_name}_evaluated.npz.\n')

        labelled_dataset = []
        for el in evaluated_dataset:
            if 'dft_energy' in el and 'dft_forces' in el:
                labelled_dataset.append(el)

        if len(labelled_dataset) > 0:
            logging.info(f'Calculating global RMSE for dataset {jj+1}...')
            # compute global RMSE
            
            RMSE = {}
            not_splited_list = []
            for el in labelled_dataset:
                if 'set' not in el:
                    not_splited_list.append(el)
                elif el['set'] not in ['TRAINING', 'TEST', 'VALIDATION']:
                    not_splited_list.append(el)

            if len(not_splited_list) > 0:
                RMSE['NOT_SPLITTED'] = global_rmse(not_splited_list)
            if len(not_splited_list) < len(labelled_dataset):
                RMSE['TRAINING'] = global_rmse([el for el in labelled_dataset if el['set'] == 'TRAINING'])
                RMSE['VALIDATION'] = global_rmse([el for el in labelled_dataset if el['set'] == 'VALIDATION'])
                RMSE['TEST'] = global_rmse([el for el in labelled_dataset if el['set'] == 'TEST'])
            
            RMSE['ALL'] = global_rmse(labelled_dataset)

            logging.info("Error-table:\n" + str(rmse_table(RMSE)))
            logging.info(f'Saving global RMSE for dataset {jj+1}...')
            np.savez(f'{dataset_name}_rmse.npz', rmse = RMSE)
        else:
            logging.info(f'No Labelled structures found in dataset {jj+1}. Skipping RMSE calculation.')
    logging.info('DONE!')


def set_logger(level=logging.INFO):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) 
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(ch)


if __name__ == "__main__":
    set_logger()
    main()