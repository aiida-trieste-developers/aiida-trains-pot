# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, ToContext, calcfunction, append_, launch, while_
from aiida import load_profile
from aiida.orm import Code, Dict, Float, Str, StructureData, load_group, List, Int, Float, SinglefileData, TrajectoryData, BandsData, RemoteData, FolderData, Data
from aiida.orm.groups import Group
from aiida.tools.groups import GroupPath
from aiida.common import AttributeDict, exceptions
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin, recursive_merge
load_profile()
from random import randint
from ase.calculators.singlepoint import SinglePointCalculator


PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
StructureData = DataFactory('core.structure')
TrajectoryData = DataFactory('core.array.trajectory')
SinglefileData = DataFactory('core.singlefile')


@calcfunction
def WriteDataset(**params):
    """Calculation function to write a dataset to a file

    :param structures: A list of AiiDA `StructureData` nodes
    """
    from ase.io import write
    from aiida.orm import SinglefileData
    import os

    dataset_list = []
    for key, value in params.items():

        en = value['out_params'].dict.energy
        tr = value['out_trajectory']
        dataset_list.append({'energy': Float(en),
                             'cell': List(list(tr.get_cells()[0])),
                             'symbols': List(list(tr.symbols)),
                             'positions': List(list(tr.get_array('positions')[0])), 
                             'forces': List(list(tr.get_array('forces')[0])),
                             'stress': List(list(tr.get_array('stress')[0])),
                             'input_structure_pk': Int(value['in_structure'].pk), 
                             })

    return {'dataset_list':List(dataset_list)}

class QECalculationWorkChain(ProtocolMixin, WorkChain):
    """WorkChain to generate a training dataset for a given structure using Quantum ESPRESSO."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(PwBaseWorkChain, namespace="scf", exclude=('pw.structure',), namespace_options={'validator': None})
        spec.input_namespace("structures", valid_type=StructureData)
        spec.output("dataset_list", valid_type=List)
        spec.outline(
            cls.run_rattle_qe,
            while_(cls.err_300)(cls.error_check),
            cls.error_check,
            cls.finalize,
            cls.results,
        )
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


    def _run_rattle_and_submit(self, structure):
        self.config += 1
      
        default_inputs = {'CONTROL': {'calculation': 'scf', 'tstress': True, 'tprnfor': True}}
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = structure
        inputs.metadata.call_link_label = f'scf_config_{self.config}'
        inputs.pw.parameters = Dict(recursive_merge(default_inputs, inputs.pw.parameters.get_dict()))
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        future = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launched PwBaseWorkChain for configuration {self.config} <{future.pk}>')
        self.to_context(pw_calculations=append_(future))
   

    def run_rattle_qe(self):
        """Run calculations for dataset generation."""

        self.config = 0
        for structure in [s for s in self.inputs.structures.values()]:
            self._run_rattle_and_submit(structure)
    
    def results(self):
        """Process results."""
        inputs = {}

        count = 0
        for value in self.ctx.pw_calculations:
            count += 1            
            inputs[f'conf{count}'] = {"out_params": value.base.links.get_outgoing().get_node_by_label("output_parameters"), "out_trajectory": value.base.links.get_outgoing().get_node_by_label("output_trajectory"), "in_structure": value.inputs.pw.structure}
        
        out= WriteDataset(**inputs)
        dataset_list = out['dataset_list']

        self.out("dataset_list", dataset_list)        


    def finalize(self):
        """Finalize."""

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

        self.out('output_parameters_scf', output_params_scf)
        self.out('output_trajectory_scf', output_trajectory_scf)
        self.out('output_band_scf', output_band_scf)
        self.out('remote_folder_scf', output_remote_folder_scf)
        self.out('retrieved_list_scf', output_retrieved_list_scf)


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
