# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, ToContext, calcfunction, append_, launch, while_
from aiida import load_profile
from aiida.orm import Code, Dict, Float, Str, StructureData, load_group, List, Int, Float, SinglefileData, TrajectoryData, BandsData, RemoteData, FolderData, Data
from aiida.orm.groups import Group
from aiida.tools.groups import GroupPath
from aiida.common import AttributeDict, exceptions
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
# from aiida.tools.data.array.trajectory import _get_aiida_structure_inline
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin, recursive_merge
load_profile()
from random import randint
from ase.calculators.singlepoint import SinglePointCalculator


PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
StructureData = DataFactory('core.structure')
TrajectoryData = DataFactory('core.array.trajectory')
SinglefileData = DataFactory('core.singlefile')



# @calcfunction
# def RattleGenerator(structure, rattle_radius, sigma_strain, n_vacancies=1):
#     """Calculation function to rattle a structure

#     :param structure: An AiiDA `StructureData` to rattle
#     :param rattle_radius: Float with the rattle radius
#     :param sigma_strain: Float with strain factor
#     :param n_vacancies: Int with number of vacancies to introduce
#     """
#     from aiida.orm import List
#     from aiida.plugins import DataFactory
#     from ase import Atoms
#     import numpy as np

#     StructureData = DataFactory('structure')


#     ase_structure = structure.get_ase()
#     seed = randint(1, 100000)
#     ase_structure.rattle(rattle_radius, seed=seed)
#     ase_structure.set_cell(ase_structure.get_cell() * sigma_strain, scale_atoms=True)
#     for _ in range(int(n_vacancies)):
#         rnd = randint(0, len(ase_structure.get_positions())-1)
#         del ase_structure[rnd]

#     return StructureData(ase=ase_structure, label='RattleStructure')





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




class DatasetGeneratorWorkChain(ProtocolMixin, WorkChain):
    """WorkChain to generate a training dataset for a given structure using Quantum ESPRESSO."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(PwBaseWorkChain, namespace="scf", exclude=('pw.structure',), namespace_options={'validator': None})
        # spec.input("code", valid_type=Code)
        # spec.input("pseudo_family_label", valid_type=Str)
        spec.input_namespace("structures", valid_type=StructureData)
        #spec.input("rattle_params", valid_type=Dict)
        spec.output("xyz_dataset", valid_type=SinglefileData)
        spec.output("dataset_list", valid_type=List)
        spec.outline(
            # cls.setup,
            #cls.check_inputs,
            cls.run_rattle_qe,
            while_(cls.err_300)(cls.error_check),
            cls.error_check,
            cls.finalize,
            cls.results,
        )
        # spec.expose_outputs(PwBaseWorkChain, namespace="scf")
        # spec.output_namespace("scf", valid_type=Dict)
        spec.output_namespace('output_parameters_scf', valid_type=Dict, dynamic=True, help='The output parameters of each ``PwBaseWorkChain`` performed``.')
        spec.output_namespace('output_trajectory_scf', valid_type=TrajectoryData, dynamic=True, help='The output trajectory of each ``PwBaseWorkChain`` performed.')
        spec.output_namespace('output_band_scf', valid_type=BandsData, dynamic=True, help='The output bands of each ``PwBaseWorkChain`` performed.')
        spec.output_namespace('remote_folder_scf', valid_type=RemoteData, dynamic=True, help='The remote folder of each ``PwBaseWorkChain`` performed.')
        spec.output_namespace('retrieved_list_scf', valid_type=FolderData, dynamic=True, help='The retrieved folder of each ``PwBaseWorkChain`` performed.')


    @classmethod
    def get_builder_from_protocol(
        cls, code, structure_list, pseudos=None, core_hole_treatments=None, protocol=None,
        overrides=None, elements_list=None, atoms_list=None, options=None,
        structure_preparation_settings=None, correction_energies=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param pseudos: the core-hole pseudopotential pairs (ground-state and
                        excited-state) for the elements to be calculated. These must
                        use the mapping of {"element" : {"core_hole" : <upf>, "gipaw" : <upf>}}
        :param protocol: the protocol to use. If not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the
                          XpsWorkChain itself.
        :param kwargs: additional keyword arguments that will be passed to the
            ``get_builder_from_protocol`` of all the sub processes that are called by this
            workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        pw_args = (code, structure_list[0], protocol)
        scf = PwBaseWorkChain.get_builder_from_protocol(
            *pw_args, overrides = None, options=options, **kwargs
        )
        
        builder = cls.get_builder()
        builder.scf = scf
        builder.structures = {f's{ii}':s for ii, s in enumerate(structure_list)} #structures[0]#List([structures[0]])

        return builder

    # def setup(self):
    #     """Set up calculations."""
    #     # super().setup()
    #     self.inputs = self.exposed_inputs(PwBaseWorkChain, 'scf')
    #     # self.inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'pwchain'))


    def _run_rattle_and_submit(self, structure, rattle_radius=0.0, sigma_strain=1.0, n_vacancies=0):
        self.config += 1
        #mod_structure = RattleGenerator(structure, rattle_radius, sigma_strain, n_vacancies=n_vacancies)
        
        default_inputs = {'CONTROL': {'calculation': 'scf', 'tstress': True, 'tprnfor': True}}

        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        #inputs.pw.structure = mod_structure
        inputs.pw.structure = structure
        # inputs.pw.pseudos = load_group('SSSP/1.2/PBE/efficiency')
        inputs.metadata.call_link_label = f'scf_config_{self.config}'
        inputs.pw.parameters = Dict(recursive_merge(default_inputs, inputs.pw.parameters.get_dict()))
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        future = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launched PwBaseWorkChain for configuration {self.config} <{future.pk}>')
        self.to_context(pw_calculations=append_(future))

    # def check_inputs(self):
    #     """Check inputs."""
    #     rattle_params = self.inputs.rattle_params.get_dict()
    #     rattle_radius_list = rattle_params['rattle_radius_list']
    #     sigma_strain_list = rattle_params['sigma_strain_list']
    #     n_configs = rattle_params['n_configs']
    #     frac_vacancies = rattle_params['frac_vacancies']
    #     vacancies_per_config = rattle_params['vacancies_per_config']

    #     for r in rattle_radius_list:
    #         if r < 0.0:
    #             raise ValueError('rattle_radius must be non-negative')
    #     for s in sigma_strain_list:
    #         if s <= 0.0:
    #             raise ValueError('sigma_strain must be positive')
    #         elif s > 1.5:
    #             raise Warning('sigma_strain is greater than 1.5')
    #         elif s < 0.5:
    #             raise Warning('sigma_strain is less than 0.5')
    #     if n_configs < 1:
    #         raise ValueError('n_configs must be at least 1')
    #     if frac_vacancies < 0.0 or frac_vacancies > 1.0:
    #         raise ValueError('frac_vacancies must be between 0 and 1')
    #     if vacancies_per_config < 0:
    #         raise ValueError('vacancies_per_config must be non-negative')
        

    def run_rattle_qe(self):
        """Run calculations for dataset generation."""


        # rattle_params = self.inputs.rattle_params.get_dict()
        # rattle_radius_list = rattle_params['rattle_radius_list']
        # sigma_strain_list = rattle_params['sigma_strain_list']
        # n_configs = rattle_params['n_configs']
        # frac_vacancies = rattle_params['frac_vacancies']
        # vacancies_per_config = rattle_params['vacancies_per_config']
        # if 'do_equilibrium' in rattle_params:
        #     do_equilibrium = rattle_params['do_equilibrium']
        # else:
        #     do_equilibrium = True

        self.config = 0
        
        # for structure in [s for s in self.inputs.structures.values()]:
        #     # self.config += 1
        #     equilibrium_calculation = False

        #     for r in rattle_radius_list:
        #         for s in sigma_strain_list:
        #             for i in range(int(n_configs)):
        #                 # config += 1
        #                 if i > int(n_configs) * frac_vacancies:
        #                     n_vacancies = vacancies_per_config
        #                 else:
        #                     n_vacancies = 0

        #                 self._run_rattle_and_submit(structure, r, s, n_vacancies)
                        
        #                 if r == 0.0 and s == 1.0 and n_vacancies == 0:
        #                     equilibrium_calculation = True

        #     if not equilibrium_calculation and do_equilibrium:
        #         self._run_rattle_and_submit(structure, 0.0, 1.0, 0)
        for structure in [s for s in self.inputs.structures.values()]:
            self._run_rattle_and_submit(structure, 0.0, 1.0, 0)


    
    def results(self):


        """Process results."""
        inputs = {}

        count = 0
        for value in self.ctx.pw_calculations:
            count += 1
            try:
                rattle_radius = value.inputs.pw__structure.get_incoming().get_node_by_label('result').inputs.rattle_radius
                sigma_strain = value.inputs.pw__structure.get_incoming().get_node_by_label('result').inputs.sigma_strain
                n_vacancies = value.inputs.pw__structure.get_incoming().get_node_by_label('result').inputs.n_vacancies
            except:
                # rattle_radius = 0.0
                # sigma_strain = 0.0
                # n_vacancies = 0
                pass

            inputs[f'conf{count}'] = {"out_params": value.get_outgoing().get_node_by_label("output_parameters"),
                             "out_trajectory": value.get_outgoing().get_node_by_label("output_trajectory"),
                             "rattle_radius": rattle_radius,
                             "sigma_strain": sigma_strain,
                             "n_vacancies": n_vacancies,
                             "in_structure": value.inputs.pw__structure.get_incoming().get_node_by_label('result').inputs.structure}
        # inputs = {label: value for label, value in self.ctx.items()}
        
        out= WriteDataset(**inputs)
        dataset_file = out['dataset_file']
        dataset_list = out['dataset_list']

        self.out("xyz_dataset", dataset_file)
        self.out("dataset_list", dataset_list)
        


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



    def error_check(self):
        """Check for errors."""
        for ii, calc in enumerate(self.ctx.pw_calculations): 
            if calc.exit_status != 0:
                if calc.exit_status in [300]:
                    restart_builder = calc.get_builder_restart()
                    restart_builder.scf.pw.metadata.options.resources['num_machines']=4
                    _, node = launch.run.get_node(restart_builder)
                    self.report(f'restarting calculation {calc.pk} <{calc.uuid}>')
                    self.ctx.pw_calculations[ii] = node
                else:
                    self.ctx.pw_calculations.pop(ii)

    def err_300(self):
        """Check for errors."""
        for ii, calc in enumerate(self.ctx.pw_calculations): 
            if calc.exit_status in [300]:
                return True
        return False
