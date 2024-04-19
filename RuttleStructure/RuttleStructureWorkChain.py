# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, ToContext, calcfunction, append_, launch, while_
from aiida import load_profile
from aiida.orm import load_node, Code, Dict, Float, Str, StructureData, load_group, List, Int, Float, SinglefileData, TrajectoryData, BandsData, RemoteData, FolderData, Data
from aiida.orm.groups import Group
from aiida.tools.groups import GroupPath
from aiida.common import AttributeDict, exceptions
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory

load_profile()
from random import randint
from ase.calculators.singlepoint import SinglePointCalculator



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

    structures_parameters_list = []  
    structures_list = []   
    for key, value in params.items():    
    	structures_parameters_list.append({'rattle_radius': Float(value['rattle_radius'].value), 'sigma_strain': Float(value['sigma_strain'].value), 'n_vacancies': Int(value['n_vacancies'].value), 'out_structure_pk': Int(value['out_structures'].pk), })
	#structures_list.append({'out_structure': value['out_structures'], })
    	#TODO
    	#atm.set_calculator(SinglePointCalculator(atm, energy=en, forces=tr.get_array('forces')[0], stress=stress))
    	atm = value['out_structures'].get_ase()
    	write(f'structures.xyz', atm, format='extxyz', append=True)

    structures_file=SinglefileData(file=f'{os.path.abspath(os.getcwd())}/structures.xyz')

    os.remove(f'{os.path.abspath(os.getcwd())}/structures.xyz')

    return {'structures_file': structures_file, 'structures_parameters_list': List(structures_parameters_list)}

#    return {'structures_list': List(structures_list)}


class RuttleStructureWorkChain(WorkChain):
    """WorkChain to generate a training dataset for a given structure using Quantum ESPRESSO."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        #spec.input_namespace("structures", valid_type=StructureData)
        #spec.input("structures", valid_type=StructureData)
        spec.input("structure_uuids", valid_type=List)        
        spec.input("rattle_params", valid_type=Dict)
        spec.output("structures_file", valid_type=SinglefileData)
        #spec.output("structures_list", valid_type=List)
        spec.output("structures_parameters_list", valid_type=List)
        
        spec.outline(
            cls.check_inputs,
            cls.run_rattle,
            #cls.finalize,
            cls.results,
        )

    def check_inputs(self):
        """Check inputs."""
        rattle_params = self.inputs.rattle_params.get_dict()
        rattle_radius_list = rattle_params['rattle_radius_list']
        sigma_strain_list = rattle_params['sigma_strain_list']
        n_configs = rattle_params['n_configs']
        frac_vacancies = rattle_params['frac_vacancies']
        vacancies_per_config = rattle_params['vacancies_per_config']

        for r in rattle_radius_list:
            if r < 0.0:
                raise ValueError('rattle_radius must be non-negative')
        for s in sigma_strain_list:
            if s <= 0.0:
                raise ValueError('sigma_strain must be positive')
            elif s > 1.5:
                raise Warning('sigma_strain is greater than 1.5')
            elif s < 0.5:
                raise Warning('sigma_strain is less than 0.5')
        if n_configs < 1:
            raise ValueError('n_configs must be at least 1')
        if frac_vacancies < 0.0 or frac_vacancies > 1.0:
            raise ValueError('frac_vacancies must be between 0 and 1')
        if vacancies_per_config < 0:
            raise ValueError('vacancies_per_config must be non-negative')
        
   
       
    def run_rattle(self):
         """Run calculations for dataset generation."""
		
         #print(self)
         rattle_params = self.inputs.rattle_params.get_dict()
         rattle_radius_list = rattle_params['rattle_radius_list']
         sigma_strain_list = rattle_params['sigma_strain_list']
         n_configs = rattle_params['n_configs']
         frac_vacancies = rattle_params['frac_vacancies']
         vacancies_per_config = rattle_params['vacancies_per_config']
         if 'do_equilibrium' in rattle_params:
             do_equilibrium = rattle_params['do_equilibrium']
         else:
             do_equilibrium = True

         self.config = 0
        
         #for structure in [s for s in self.inputs.structures.values()]:
         for structure_uuid in self.inputs.structure_uuids:
             structure = load_node(structure_uuid)
             equilibrium_calculation = False

             for r in rattle_radius_list:
                 for s in sigma_strain_list:
                     for i in range(int(n_configs)):
                         if i > int(n_configs) * frac_vacancies:
                             n_vacancies = vacancies_per_config
                         else:
                             n_vacancies = 0

                         self._run_rattle(structure, r, s, n_vacancies)
                        
                         if r == 0.0 and s == 1.0 and n_vacancies == 0:
                             equilibrium_calculation = True

             if not equilibrium_calculation and do_equilibrium:
                 self._run_rattle(structure, 0.0, 1.0, 0)
        
        
    def _run_rattle(self, structure, rattle_radius=0.0, sigma_strain=1.0, n_vacancies=0):
        self.config += 1
        mod_structure = RattleGenerator(structure, rattle_radius, sigma_strain, n_vacancies=n_vacancies)
        
        

        inputs = AttributeDict()
        inputs.config = self.config
        inputs.structure = mod_structure
        inputs.rattle_radius = rattle_radius
        inputs.sigma_strain = sigma_strain
        inputs.n_vacancies = n_vacancies
                        
        #self.to_context(mod_structures=append_(inputs))
        self.ctx.mod_structures = [inputs]

    def results(self):

        """Process results."""
        inputs = {}

        count = 0
        for value in self.ctx.mod_structures:
            count += 1
            try:
                rattle_radius = value.rattle_radius
                sigma_strain = value.sigma_strain
                n_vacancies = value.n_vacancies
            except:
                # rattle_radius = 0.0
                # sigma_strain = 0.0
                # n_vacancies = 0
                pass

            inputs[f'conf{count}'] = {"rattle_radius": Float(rattle_radius),
                             "sigma_strain": Float(sigma_strain),
                             "n_vacancies": Int(n_vacancies),
                             "out_structures": value.structure}
        
        out= WriteDataset(**inputs)
        structures_file = out['structures_file']
        #structures_list = out['structures_list']
        structures_parameters_list = out['structures_parameters_list']

        self.out("structures_file", structures_file)
        
        # Add structures_list and structures_parameters_list only if they are not already present
        # Add structures_list and structures_parameters_list only if they are not already present
        #if "structures_list" not in self.exposed_outputs:
        #	self.out("structures_list", structures_list)

        #if "structures_parameters_list" not in self.exposed_outputs:
        #	self.out("structures_parameters_list", structures_parameters_list)


        #self.out("structures_list", structures_list)
        self.out("structures_parameters_list", structures_parameters_list)
        


    def finalize(self):
        """Finalize."""

        # count = 0
        # labels = []
        # for val in self.ctx.pw_calculations:
        #     count += 1
        #     labels.append(f'config_{count}')
        output_params_scf = {}
        output_trajectory_scf = {}
        output_band_scf = {}
        output_remote_folder_scf = {}
        output_retrieved_list_scf = {}

        for ii, val in enumerate(self.ctx.pw_calculations):
            output_params_scf[f'config_{ii+1}'] = val.outputs.output_parameters
            output_trajectory_scf[f'config_{ii+1}'] = val.outputs.output_trajectory
            output_band_scf[f'config_{ii+1}'] = val.outputs.output_band
            output_remote_folder_scf[f'config_{ii+1}'] = val.outputs.remote_folder
            output_retrieved_list_scf[f'config_{ii+1}'] = val.outputs.retrieved

        # labels = list(self.ctx.pw_calculations.keys())
        # output_params_scf = {label : self.ctx[label].outputs.output_parameters for label in self.ctx.pw_calculations}
        self.out('output_parameters_scf', output_params_scf)
        self.out('output_trajectory_scf', output_trajectory_scf)
        self.out('output_band_scf', output_band_scf)
        self.out('remote_folder_scf', output_remote_folder_scf)
        self.out('retrieved_list_scf', output_retrieved_list_scf)
        # for val in self.ctx.pw_calculations:
        #     # self.report(f'val: {val}')
        #     count += 1
        #     # if count == 1:
        #     #     self.out_many(self.exposed_outputs(val, PwBaseWorkChain))
        #     # else:
        #     self.out_many(self.exposed_outputs(val, PwBaseWorkChain, namespace=f'scf', agglomerate=False))




