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

def get_parity_data(dataset):
    """Get data for parity plots from dataset
    
    Parameters
    ----------
    dataset : list
        List of dictionaries containing the information of the
        evaluated dataset.

    Returns
    -------
    dict
        Dictionary containing the DFT and DNN quantities for
        energy, forces and stress.
    """
    if len(dataset) == 0:
        return {
            'dft_e': np.array([]),
            'pot_e': np.array([]),
            'std_pot_e': np.array([]),
            'dft_f': np.array([[],[],[]]).T,
            'pot_f': np.array([[],[],[]]).T,
            'std_pot_f': np.array([[],[],[]]).T,
            'dft_s': np.array([]),
            'pot_s': np.array([]),
            'std_pot_s': np.array([]),
        }
    pot_e = []
    dft_e = []
    std_pot_e = []
    pot_f = []
    dft_f = []
    std_pot_f = []
    pot_s = []
    dft_s = []
    std_pot_s = []

    for frame in dataset:
        pot_e.append([frame[key]/len(frame['positions']) for key in frame.keys() if 'pot' in key and 'energy' in key and not 'rmse' in key])
        std_pot_e.append(np.std(pot_e[-1]))
        pot_e[-1] = np.mean(pot_e[-1])
        dft_e.append(frame['dft_energy']/len(frame['positions']))
        
        pot_f.append([frame[key] for key in frame.keys() if 'pot' in key and 'forces' in key and not 'rmse' in key])
        std_pot_f.append(np.std(pot_f[-1], axis=0))
        pot_f[-1] = np.mean(pot_f[-1], axis=0)
        dft_f.append(frame['dft_forces'])

        pot_s.append([frame[key] for key in frame.keys() if 'pot' in key and 'stress' in key and not 'rmse' in key])
        std_pot_s.append(np.std(pot_s[-1], axis=0))
        pot_s[-1] = np.mean(pot_s[-1], axis=0)
        dft_s.append(frame['dft_stress'])

    dft_f = np.concatenate(dft_f)
    pot_f = np.concatenate(pot_f)
    std_pot_f = np.concatenate(std_pot_f)
    dft_s = np.concatenate(dft_s)
    pot_s = np.concatenate(pot_s).ravel()
    std_pot_s = np.concatenate(std_pot_s).ravel()
    
    return {
        'dft_e': np.array(dft_e),
        'pot_e': np.array(pot_e),
        'std_pot_e': np.array(std_pot_e),
        'dft_f': np.array(dft_f),
        'pot_f': np.array(pot_f),
        'std_pot_f': np.array(std_pot_f),
        'dft_s': np.array(dft_s),
        'pot_s': np.array(pot_s),
        'std_pot_s': np.array(std_pot_s),
    }

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
    RMSE = {}
    if len(dataset) == 0:
        return RMSE
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
                dnn_s[-1].extend(frame[key].ravel())
        elif 'dft_energy' in key:
            for frame in dataset:
                dft_e.append(frame[key]/len(frame['positions']))
        elif 'dft_forces' in key:
            for frame in dataset:
                dft_f.extend(frame[key].ravel())
        elif 'dft_stress' in key:
            for frame in dataset:
                dft_s.extend(frame[key].ravel())
    
    rmse_e = np.array([calc_rmse(dft_e, dnn) for dnn in dnn_e])
    rmse_f = np.array([calc_rmse(dft_f, dnn) for dnn in dnn_f])
    rmse_s = np.array([calc_rmse(dft_s, dnn) for dnn in dnn_s])

    
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
        return np.mean(np.mean(np.sqrt(np.mean((data - mean[np.newaxis, :, :]) ** 2, axis=0))))


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
    datasets = glob.glob('dataset*xyz')

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
            evaluated_dataset.append({k:v for k, v in atm.info.items() if k != 'energy' and k != 'stress'})
            evaluated_dataset[-1]['cell'] = np.array(atm.get_cell())
            evaluated_dataset[-1]['positions'] = np.array(atm.get_positions())
            evaluated_dataset[-1]['symbols'] = atm.get_chemical_symbols()
            evaluated_dataset[-1]['pbc'] = atm.get_pbc()
            try:
                evaluated_dataset[-1]['dft_energy'] = atm.get_potential_energy()
                evaluated_dataset[-1]['dft_stress'] = np.array(atm.get_stress(voigt=False).ravel())
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
                    evaluated_dataset[-1][f'pot_{n_pot}_energy_rmse'] = calc_rmse([evaluated_dataset[-1]['dft_energy']], [evaluated_dataset[-1][f'pot_{n_pot}_energy']])
                if 'dft_forces' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'pot_{n_pot}_forces_rmse'] = calc_rmse(evaluated_dataset[-1]['dft_forces'], evaluated_dataset[-1][f'pot_{n_pot}_forces'])
                if 'dft_stress' in evaluated_dataset[-1]:
                    evaluated_dataset[-1][f'pot_{n_pot}_stress_rmse'] = calc_rmse(evaluated_dataset[-1]['dft_stress'], np.array(atm.get_stress(voigt=False)).ravel())
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
            PARITY = {}
            not_splited_list = []
            for el in labelled_dataset:
                if 'set' not in el:
                    not_splited_list.append(el)
                elif el['set'] not in ['TRAINING', 'TEST', 'VALIDATION']:
                    not_splited_list.append(el)

            if len(not_splited_list) > 0:
                RMSE['NOT_SPLITTED'] = global_rmse(not_splited_list)
                PARITY.update({f'not_splitted_{k}':v for k, v in get_parity_data(not_splited_list).items()})
            if len(not_splited_list) < len(labelled_dataset):
                training_dataset = [el for el in labelled_dataset if el['set'] == 'TRAINING']
                validation_dataset = [el for el in labelled_dataset if el['set'] == 'VALIDATION']
                test_dataset = [el for el in labelled_dataset if el['set'] == 'TEST']
                RMSE['TRAINING'] = global_rmse(training_dataset)
                RMSE['VALIDATION'] = global_rmse(validation_dataset)
                RMSE['TEST'] = global_rmse(test_dataset)
                PARITY.update({f'training_{k}':v for k, v in get_parity_data(training_dataset).items()})
                PARITY.update({f'validation_{k}':v for k, v in get_parity_data(validation_dataset).items()})
                PARITY.update({f'test_{k}':v for k, v in get_parity_data(test_dataset).items()})
            
            RMSE['ALL'] = global_rmse(labelled_dataset)

            logging.info("Error-table:\n" + str(rmse_table(RMSE)))
            logging.info(f'Saving global RMSE for dataset {jj+1}...')
            np.savez(f'{dataset_name}_rmse.npz', rmse = RMSE)
            logging.info(f'Saving data for parity plots for dataset {jj+1}...')
            np.savez(f'{dataset_name}_parity.npz', parity = PARITY)
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