# -*- coding: utf-8 -*-
""" DataGenerationWorkChain to generate a training dataset """
from aiida.engine import WorkChain, calcfunction
from aiida import load_profile
from aiida.orm import Bool, Float, Str, StructureData, List, Int, Float
from aiida.plugins import DataFactory
import numba
from random import randint, uniform
from ase import Atoms
import numpy as np

load_profile()


StructureData = DataFactory('core.structure')
SinglefileData = DataFactory('core.singlefile')


@calcfunction
def RattleStructureGenerator(n_configs, rattle_fraction, max_sigma_strain, frac_vacancies, vacancies_per_config, **in_structure_dict):
    """Generate structures.
    
    :param in_structure_list: A list of AiiDA `StructureData` nodes
    :param n_configs: Int with the number of configurations to generate
    :param rattle_fraction: Float with the rattle fraction
    :param max_sigma_strain: Float with the maximum strain factor
    :param frac_vacancies: Float with the fraction of vacancies
    :param vacancies_per_config: Int with the number of vacancies per configuration
    """
    
    structure_list = []
    for _, structure in in_structure_dict.items():
        ase_structure = structure.get_ase()
        min_interatomic_distances = get_min_interatomic_distances(ase_structure.get_positions(), np.array(ase_structure.get_cell()))

        
        for i in range(int(n_configs)):
            if i > int(n_configs) * frac_vacancies:
                n_vacancies = vacancies_per_config.value
            else:
                n_vacancies = 0

            mod_structure = ase_structure.copy()
            sigma_strain = uniform(1-max_sigma_strain.value, 1+max_sigma_strain.value)
            mod_structure.set_cell(ase_structure.get_cell() * sigma_strain, scale_atoms=True)
            mod_structure.set_positions(uniform_random_atomic_displacement(mod_structure.get_positions(), min_interatomic_distances*sigma_strain, rattle_fraction.value))
            for _ in range(int(n_vacancies)):
                rnd = randint(0, len(mod_structure.get_positions())-1)
                del mod_structure[rnd]
        
            structure_list.append({'cell': List(list(mod_structure.get_cell())),
                    'symbols': List(list(mod_structure.get_chemical_symbols())),
                    'positions': List(list(mod_structure.get_positions())), 
                    'rattle_fraction': rattle_fraction,
                    'max_sigma_strain': max_sigma_strain,
                    'n_vacancies': n_vacancies,
                    'input_structure_uuid': Str(structure.uuid),
                    'gen_method': Str('RATTLE')
                    })
    
    return {'rattle_structure_list': List(structure_list)}

@calcfunction
def InputStructureGenerator(**in_structure_dict):
    """Add input structures to the dataset.

    :param in_structure_list: List of AiiDA `StructureData` nodes
    """
    structure_list = []
    for _, structure in in_structure_dict.items():
        ase_structure = structure.get_ase()
        structure_list.append({'cell': List(list(ase_structure.get_cell())),
                    'symbols': List(list(ase_structure.get_chemical_symbols())),
                    'positions': List(list(ase_structure.get_positions())), 
                    'input_structure_uuid': Str(structure.uuid),
                    'gen_method': Str('INPUT_STRUCTURE')
                    })
        
    return {'input_structure_list': List(structure_list)}

@calcfunction
def IsolatedStructureGenerator(**in_structure_dict):
    """Generate isolated atoms.
    
    :param in_structure_list: List of AiiDA `StructureData` nodes
    """
        
    structure_list = []
    done_types = []
    for _, structure in in_structure_dict.items():
        for atm_type in list(structure.get_symbols_set()):
            if atm_type not in done_types:
                done_types.append(atm_type)
                isolated_structure = Atoms(atm_type, positions=[[0.0, 0.0, 0.0]], cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
                
                structure_list.append({'cell': List(list(isolated_structure.get_cell())),
                    'symbols': List(list(isolated_structure.get_chemical_symbols())),
                    'positions': List(list(isolated_structure.get_positions())), 
                    'gen_method': Str('ISOLATED_ATOM')
                    })

    return {'isolated_atoms_structure_list': List(structure_list)}
    
            


@calcfunction
def WriteDataset(**dataset_lists_dict):
    """Combine datasets to single dataset_list.

    :param structures: A list of AiiDA `StructureData` nodes
    """

    dataset_out_list = []
    for _, dataset in dataset_lists_dict.items():
        dataset_out_list.extend(dataset)

    return {'global_structure_list':List(dataset_out_list)}


@numba.njit(parallel=True)
def get_min_interatomic_distances(positions, cell):
    """For each atom, calculate the minimum distance to any other atom in the structure.

    :param positions: A numpy array of atomic positions
    :param cell: A numpy array of the cell vectors
    """
    
    N_P, _ = positions.shape
    N_C, _ = cell.shape
    min_dist = np.zeros((N_P))
    dist = np.zeros((N_P, N_P))
    for ii in numba.prange(N_P):
        for jj in numba.prange(N_P):
                hidden_dist = np.zeros((N_C, N_C, N_C))
                for i in numba.prange(-1, 2):
                    for j in numba.prange(-1, 2):
                        for k in numba.prange(-1, 2):
                            for l in numba.prange(N_C):
                                hidden_dist[i,j,k] += (positions[ii,l] - positions[jj,l] + i*cell[0,l] + j*cell[1,l] + k*cell[2,l])**2
                            hidden_dist[i,j,k] = np.sqrt(hidden_dist[i,j,k])
                dist[ii,jj] = np.min(hidden_dist)
                if ii == jj:
                    dist[ii,jj] = np.inf
        min_dist[ii] = np.min(dist[ii,:])
    return min_dist

@numba.njit(parallel=True)
def uniform_random_atomic_displacement(positions, min_distances, max_displacement_fraction):
    """Displace atoms randomly in a uniform manner.

    :param positions: A numpy array of atomic positions
    :param min_distances: A numpy array of minimum interatomic distances
    :param max_displacement_fraction: A float that determines the maximum displacement as a fraction of the minimum interatomic distance
    """
    
    N_P, _ = positions.shape
    for ii in numba.prange(N_P):
            rand_dir = np.array([uniform(0,1),uniform(0,1),uniform(0,1)])
            rand_dir /= np.sqrt(rand_dir[0]**2+rand_dir[1]**2+rand_dir[2]**2)
            positions[ii] += uniform(0,1)*min_distances[ii]*max_displacement_fraction*rand_dir
    return positions


class DataGenerationWorkChain(WorkChain):
    """WorkChain to generate a training dataset."""


   ######################################################
   ##                 DEFAULT VALUES                   ##
   ######################################################
    DEFAULT_RATTLE_rattle_fraction          = Float(0.1)
    DEFAULT_RATTLE_max_sigma_strain         = Float(0.1)
    DEFAULT_RATTLE_n_configs                = Int(50)
    DEFAULT_RATTLE_frac_vacancies           = Float(0.4)
    DEFAULT_RATTLE_vacancies_per_config     = Int(2)

    DEFAULT_do_rattle                       = Bool(True)
    DEFAULT_do_input                        = Bool(True)
    DEFAULT_do_isolated                     = Bool(True)
   ######################################################



    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input_namespace("structures", valid_type=StructureData, required=True)

        spec.input("do_rattle", valid_type=Bool, default=lambda:cls.DEFAULT_do_rattle, required=False, help=f"Perform rattle calculations (random atomic displacements, cell stretch/compression, vacancies. Permutations and replacements are not yet implemented). Default: {cls.DEFAULT_do_rattle}")
        spec.input("do_input", valid_type=Bool, default=lambda:cls.DEFAULT_do_input, required=False, help=f"Add input structures to the dataset. Default: {cls.DEFAULT_do_input}")
        spec.input("do_isolated", valid_type=Bool, default=lambda:cls.DEFAULT_do_isolated, required=False, help=f"Add isolated atoms configurations to the dataset. Default: {cls.DEFAULT_do_isolated}")


        spec.input("rattle.params.rattle_fraction", valid_type=(Int,Float), default=lambda:cls.DEFAULT_RATTLE_rattle_fraction, required=False, help=f"Atoms are displaced by a rattle_fraction of the minimum interatomic distance. Default: {cls.DEFAULT_RATTLE_rattle_fraction}")
        spec.input("rattle.params.max_sigma_strain", valid_type=(Int,Float), default=lambda:cls.DEFAULT_RATTLE_max_sigma_strain, required=False, help=f"Maximum strain factor. Default: {cls.DEFAULT_RATTLE_max_sigma_strain}")
        spec.input("rattle.params.n_configs", valid_type=Int, default=lambda:cls.DEFAULT_RATTLE_n_configs, required=False, help=f"Number of configurations to generate. Default: {cls.DEFAULT_RATTLE_n_configs}")
        spec.input("rattle.params.frac_vacancies", valid_type=(Int,Float), default=lambda:cls.DEFAULT_RATTLE_frac_vacancies, required=False, help=f"Fraction of configurations with vacancies. Default: {cls.DEFAULT_RATTLE_frac_vacancies}")
        spec.input("rattle.params.vacancies_per_config", valid_type=Int, default=lambda:cls.DEFAULT_RATTLE_vacancies_per_config, required=False, help=f"Number of vacancies per configuration. Default: {cls.DEFAULT_RATTLE_vacancies_per_config}")

        spec.output_namespace("structure_lists", valid_type=List, dynamic=True)

        
        spec.outline(
            cls.check_inputs,
            cls.run_dataset_generation
        )

    @classmethod
    def get_builder_with_structures(cls, structures):
        """Return a builder"""
        
        builder = cls.get_builder()
        builder.structures = {f's{ii}':s for ii, s in enumerate(structures)}
        return builder

    def check_inputs(self):
        """Check inputs."""
        
        if self.inputs.do_rattle:
            # ERRORS
            if self.inputs.rattle.params.rattle_fraction < 0.0 or self.inputs.rattle.params.rattle_fraction > 1.0:
                raise ValueError('rattle_fraction must be between 0 and 1')
            if self.inputs.rattle.params.max_sigma_strain < 0.0 or self.inputs.rattle.params.max_sigma_strain > 1.0:
                raise ValueError('max_sigma_strain must be between 0 and 1')
            if self.inputs.rattle.params.n_configs < 1:
                raise ValueError('n_configs must be at least 1')
            if self.inputs.rattle.params.frac_vacancies < 0.0 or self.inputs.rattle.params.frac_vacancies > 1.0:
                raise ValueError('frac_vacancies must be between 0 and 1')
            if self.inputs.rattle.params.vacancies_per_config < 0:
                raise ValueError('vacancies_per_config must be non-negative')
            for structure in self.inputs.structures.values():
                if self.inputs.rattle.params.vacancies_per_config > len(structure.get_ase().get_chemical_symbols()):
                    raise ValueError(f'Number of vacancies per configuration is greater than the number of atoms in the structure <{structure.uuid}>.')
            # WARNINGS
            if self.inputs.rattle.params.rattle_fraction > 0.15:
                raise Warning('rattle_fraction is greater than 0.15 (15%), can lead to atoms too close to each other.')
            if self.inputs.rattle.params.max_sigma_strain > 0.15:
                raise Warning('max_sigma_strain is greater than 0.15 (15%), can lead to atoms too close to each other.')
            
       
    def run_dataset_generation(self):
        """Generate datasets."""

        self.report(self.inputs.structures)
        dataset_lists = {}
        if self.inputs.do_input:
            dataset_lists['input_structure_list'] = InputStructureGenerator(**dict(self.inputs.structures))['input_structure_list']
        if self.inputs.do_isolated:
            dataset_lists['isolated_atoms_structure_list'] = IsolatedStructureGenerator(**dict(self.inputs.structures))['isolated_atoms_structure_list']
        if self.inputs.do_rattle:
            dataset_lists['rattle_structure_list'] = RattleStructureGenerator(self.inputs.rattle.params.n_configs, self.inputs.rattle.params.rattle_fraction, self.inputs.rattle.params.max_sigma_strain, self.inputs.rattle.params.frac_vacancies, self.inputs.rattle.params.vacancies_per_config,**dict(self.inputs.structures))['rattle_structure_list']

        dataset_lists['global_structure_list'] = WriteDataset(**dataset_lists)['global_structure_list']
        # self.report(dataset_lists)
        self.out("structure_lists", dataset_lists)