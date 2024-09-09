# -*- coding: utf-8 -*-
"""Launch a calculation using the 'diff-tutorial' plugin"""
from pathlib import Path
import aiida
from aiida import engine, orm
from aiida.common.exceptions import NotExistent
from aiida.engine import submit
from aiida.plugins import CalculationFactory
from aiida.orm import load_node
from aiida.orm import Code, Float, Str, StructureData, Int, List, Float, SinglefileData, Bool, Dict
from aiida_lammps.data.trajectory import LammpsTrajectory

aiida.load_profile()

def LammpsFrameExtraction(correlation_time, saving_frequency, trajectory, thermalization_time=0):
    """Extract frames from trajectory."""

    
    extracted_frames = []      


    params = {}
    params = {}
    for inc in trajectory.base.links.get_incoming().all():
        if inc.node.process_type == 'aiida.calculations:lammps.base':
            lammps_id = inc.node.uuid
        if inc.node.process_type == 'aiida.workflows:lammps.base':
            for inc2 in inc.node.base.links.get_incoming().all():
                if inc2.link_label == 'lammps__parameters':
                    params = Dict(dict=inc2.node).get_dict()
                elif inc2.link_label == 'lammps__structure':
                    
                    input_structure = inc2.node.get_ase()
                    input_structure_node =  inc2.node
                    masses = []
                    symbols = []
                    symbol = input_structure.get_chemical_symbols()
                    for ii, mass in enumerate(input_structure.get_masses()):
                        if mass not in masses:
                            masses.append(mass)
                            symbols.append(symbol[ii])
                        
                    masses, symbols = zip(*sorted(zip(masses, symbols)))
    
    i = int(thermalization_time/params['control']['timestep']/saving_frequency)

    while i < trajectory.number_steps:
        step_data = trajectory.get_step_data(i)
        string_components1 = step_data[0][5].split()
        string_components2 = step_data[0][6].split()
        string_components3 = step_data[0][7].split()
        cell = [[float(value) for value in string_components1],[float(value) for value in string_components2],[float(value) for value in string_components3]]

        extracted_frames.append({'cell': List(list(cell)),
                'symbols': List(list(step_data[5]['element'])),
                'positions': List([[step_data[5]['x'][jj],step_data[5]['y'][jj],step_data[5]['z'][jj]] for jj, _ in enumerate(step_data[5]['y'])]),
                'input_structure_uuid': Str(input_structure_node.uuid),
                # 'md_forces': List(list(trajectory_frames[i].get_forces())),
                'gen_method': Str('LAMMPS')
                })
        extracted_frames[-1]['style'] = params['md']['integration']['style']
        extracted_frames[-1]['temp'] = params['md']['integration']['constraints']['temp']
        extracted_frames[-1]['timestep'] = params['control']['timestep']
        extracted_frames[-1]['id_lammps'] = lammps_id

        i = i + int(correlation_time/params['control']['timestep']/saving_frequency)

    return {'lammps_extracted_list': List(list=extracted_frames)}

correlation_time = load_node(97142)
thermalization_time =load_node(97143)
# Load the node
node = load_node(97245)
print(node.node_type)


lammps_extracted_list = LammpsFrameExtraction(correlation_time, Int(100), node, thermalization_time = thermalization_time)


print('md.lammps_extracted_list', len(lammps_extracted_list['lammps_extracted_list']))
print('md.lammps_extracted_list', lammps_extracted_list['lammps_extracted_list'][0])
