from aiida.orm import QueryBuilder, Dict, List, Int, Float
from aiida import load_profile
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import calcfunction
from ase.calculators.singlepoint import SinglePointCalculator
from random import randint

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
StructureData = DataFactory('structure')
TrajectoryData = DataFactory('core.array.trajectory')
SinglefileData = DataFactory('core.singlefile')



@calcfunction
def RattleGenerator(structure, rattle_radius, sigma_strain, n_vacancies=1):
    """Calculation function to rattle a structure

    :param structure: An AiiDA `StructureData` to rattle
    :param rattle_radius: Float with the rattle radius
    :param sigma_strain: Float with strain factor
    :param n_vacancies: Int with number of vacancies to introduce
    """
    from aiida.orm import List
    from aiida.plugins import DataFactory
    from ase import Atoms
    import numpy as np

    StructureData = DataFactory('structure')


    ase_structure = structure.get_ase()
    seed = randint(1, 100000)
    ase_structure.rattle(rattle_radius, seed=seed)
    ase_structure.set_cell(ase_structure.get_cell() * sigma_strain, scale_atoms=True)
    for _ in range(int(n_vacancies)):
        rnd = randint(0, len(ase_structure.get_positions())-1)
        del ase_structure[rnd]

    return StructureData(ase=ase_structure, label='RattleStructure')





@calcfunction
def WriteDataset(**params):
    """Calculation function to write a dataset to a file

    :param structures: A list of AiiDA `StructureData` nodes
    """
    from ase.io import write
    from aiida.orm import SinglefileData
    import os

    dataset_list = []
    # params = par.get_dict()
    for key, value in params.items():

        en = value['out_params'].dict.energy
        tr = value['out_trajectory']
        atm = tr.get_step_structure(-1).get_ase()

        dataset_list.append({'energy': Float(en),
                             'cell': List(list(tr.get_cells()[0])),
                             'symbols': List(list(tr.symbols)),
                             'positions': List(list(tr.get_array('positions')[0])), 
                             'forces': List(list(tr.get_array('forces')[0])),
                             'stress': List(list(tr.get_array('stress')[0])),
                             'rattle_radius': Float(value['rattle_radius'].value),
                             'sigma_strain': Float(value['sigma_strain'].value),
                             'n_vacancies': Int(value['n_vacancies'].value),
                             'input_structure_pk': Int(value['in_structure'].pk), 
                             })

        # en = pwchain.get_outgoing().get_node_by_label("output_parameters").dict.energy
        # tr = pwchain.get_outgoing().get_node_by_label("output_trajectory")
        
        # atm = tr.get_step_structure(-1).get_ase()
        s = tr.get_array('stress')[0]
        stress = [s[0][0] , s[1][1], s[2][2], s[1][2], s[0][2], s[0][1]]
        atm.set_calculator(SinglePointCalculator(atm, energy=en, forces=tr.get_array('forces')[0], stress=stress))

        write(f'dataset.xyz', atm, format='extxyz', append=True)

    dataset_file=SinglefileData(file=f'{os.path.abspath(os.getcwd())}/dataset.xyz')

    os.remove(f'{os.path.abspath(os.getcwd())}/dataset.xyz')

    return {'dataset_file':dataset_file, 'dataset_list':List(dataset_list)}