# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction, if_, while_
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, List, Float, SinglefileData, Bool, Dict
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.common import AttributeDict
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.lammpsrun import read_lammps_dump_text
from io import StringIO
import numpy as np
import os
import io
load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')
DatasetGeneratorWorkChain   = WorkflowFactory('NNIPdevelopment.datageneration')
PwBaseWorkChain             = WorkflowFactory('quantumespresso.pw.base')
MaceWorkChain               = WorkflowFactory('NNIPdevelopment.macetrain')
LammpsWorkChain             = WorkflowFactory('NNIPdevelopment.lammpsmd')
EvaluationCalculation       = CalculationFactory('NNIPdevelopment.evaluation')

def dataset_list_to_ase_list(dataset_list):
    """Convert dataset list to an ASE list."""

    ase_list = []

    for config in dataset_list.get_list():
        ase_list.append(Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell']))
        if 'dft_stress' in config.keys():
            s = config['stress']
            stress = [s[0][0] ,s[1][1], s[2][2], s[1][2], s[0][2], s[0][1]]
        if 'dft_energy' in config.keys() and 'dft_forces' in config.keys():
            ase_list[-1].set_calculator(SinglePointCalculator(ase_list[-1], energy=config['energy'], forces=config['forces'], stress=stress))

    return ase_list

@calcfunction
def WriteLabelledList(non_labelled_structures, **labelled_data):
    labelled_list = []
    for key, value in labelled_data.items():
        labelled_list.append(non_labelled_structures.get_list()[int(key.split('_')[1])])
        labelled_list[-1]['dft_energy'] = Float(value['output_parameters'].dict.energy)
        labelled_list[-1]['dft_forces'] = List(list(value['output_trajectory'].get_array('forces')[0]))
        labelled_list[-1]['dft_stress'] = List(list(value['output_trajectory'].get_array('stress')[0]))
    return List(list=labelled_list)

@calcfunction
def LammpsFrameExtraction(correlation_time, saving_frequency, **trajectories):
    """Extract frames from trajectory."""


    extracted_frames = []
    for _, trajectory in trajectories.items():

        params = {}
        for inc in trajectory.base.links.get_incoming().all():
            if inc.node.process_type == 'aiida.workflows:NNIPdevelopment.lammpsmd':
                for inc2 in inc.node.base.links.get_incoming().all():
                    if inc2.node.node_type in ['data.core.float.Float.', 'data.core.int.Int.']:
                        params[inc2.link_label] = inc2.node.value
                    elif inc2.node.node_type == 'data.core.structure.StructureData.':
                        
                        input_structure = inc2.node.get_ase()
                        masses = []
                        symbols = []
                        symbol = input_structure.get_chemical_symbols()
                        for ii, mass in enumerate(input_structure.get_masses()):
                            if mass not in masses:
                                masses.append(mass)
                                symbols.append(symbol[ii])
                            
                        masses, symbols = zip(*sorted(zip(masses, symbols)))
        trajectory_frames = read_lammps_dump_text(StringIO(trajectory.get_content()), index=slice(0, int(1e50), 1), specorder=list(symbols))
        
        i = 0
        while i < len(trajectory_frames):
            extracted_frames.append({'cell': List(list(trajectory_frames[i].get_cell())),
                    'symbols': List(list(trajectory_frames[i].get_chemical_symbols())),
                    'positions': List(list(trajectory_frames[i].get_positions())),
                    'md_forces': List(list(trajectory_frames[i].get_forces())),
                    'gen_method': Str('LAMMPS')
                    })
            for par, val in params.items():
                extracted_frames[-1][par] = val
            i = i + int(correlation_time/params['dt']/saving_frequency)
    for i in extracted_frames:
        print(i)
    return {'lammps_extracted_list': List(list=extracted_frames)}


class NNIPWorkChain(WorkChain):
    """WorkChain to launch LAMMPS calculations."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input('do_data_generation', valid_type=Bool, default=lambda: Bool(True), help='Do data generation', required=False)
        spec.input('do_dft', valid_type=Bool, default=lambda: Bool(True), help='Do DFT calculations', required=False)
        spec.input('do_mace', valid_type=Bool, default=lambda: Bool(True), help='Do MACE calculations', required=False)
        spec.input('do_md', valid_type=Bool, default=lambda: Bool(True), help='Do MD calculations', required=False)

        spec.input('non_labelled_list', valid_type=List, help='List of non labelled structures', required=False)
        spec.input('labelled_list', valid_type=List, help='List of labelled structures', required=False)
        spec.input('mace_workchain_pk', valid_type=Str, help='MACE workchain pk', required=False)
        spec.input('mace_lammps_potential', valid_type=SinglefileData, help='MACE potential for MD', required=False)

        spec.input('md.temperatures', valid_type=List, help='List of temperatures for MD', required=False)
        spec.input('md.pressures', valid_type=List, help='List of pressures for MD', required=False)
        spec.input('potential', valid_type=SinglefileData, help='MACE potential for MD', required=False)

        spec.input('frame_extraction.correlation_time', valid_type=Float, help='Correlation time for frame extraction', required=False)

        spec.expose_inputs(DatasetGeneratorWorkChain, namespace="datagen", exclude=('structures'))
        spec.expose_inputs(PwBaseWorkChain, namespace="dft", exclude=('pw.structure',), namespace_options={'validator': None})
        spec.expose_inputs(MaceWorkChain, namespace="mace", exclude=('dataset_list',), namespace_options={'validator': None})
        spec.expose_inputs(LammpsWorkChain, namespace="md", exclude=('structure','temperature', 'pressure', 'potential'), namespace_options={'validator': None})
        spec.expose_inputs(EvaluationCalculation, namespace="cometee_evaluation", exclude=('mace_potentials', 'datasetlist'))
        # spec.expose_inputs(FrameExtractionWorkChain, namespace="frame_extraction", exclude=('trajectories', 'input_structure', 'dt', 'saving_frequency'))

        spec.input_namespace("structures", valid_type=StructureData, required=True)

        spec.output("dft.labelled_list", valid_type=List, help="List of configurations labelled via DFT")
        spec.output("md.lammps_extracted_list", valid_type=List, help="List of extracted frames from MD trajectories")
        spec.output("cometee_evaluation_list", valid_type=List, help="List of cometee evaluated configurations")
        spec.output_namespace("md", dynamic=True, help="MD outputs")
        spec.output_namespace("mace", dynamic=True, help="MACE outputs")
        spec.expose_outputs(DatasetGeneratorWorkChain, namespace="datagen")
        # spec.expose_outputs(EvaluationCalculation, namespace="cometee_evaluation")
        # spec.expose_outputs(MaceWorkChain, namespace="mace")

        
        spec.outline(
            cls.initialization,
            if_(cls.do_data_generation)(
                cls.data_generation,
                cls.finalize_data_generation),
            while_(cls.check_iteration)(
                if_(cls.do_dft)(
                    cls.run_dft,
                    cls.finalize_dft),
                if_(cls.do_mace)(
                    cls.run_mace,
                    cls.finalize_mace),
                if_(cls.do_md)(
                    cls.run_md,
                    cls.finalize_md,
                    cls.run_md_frame_extraction,
                    cls.run_cometee_evaluation,
                    cls.finalize_cometee_evaluation),
            )
            # cls.finalize,
            # cls.save_files
        )

    @classmethod
    def get_builder_from_protocol(cls, structures, qe_code, qe_protocol=None, qe_options=None, qe_overrides=None, **kwargs):
        """Return a builder"""
        
        builder = cls.get_builder()
        builder.structures = {f's{ii}':s for ii, s in enumerate(structures)}
        builder.dft = PwBaseWorkChain.get_builder_from_protocol(*(qe_code, structures[0], qe_protocol), overrides = qe_overrides, options=qe_options, **kwargs)

        return builder
    
    def do_data_generation(self): return bool(self.inputs.do_data_generation)
    def do_dft(self): return bool(self.inputs.do_dft)
    def do_mace(self): return bool(self.inputs.do_mace)
    def do_md(self): return bool(self.inputs.do_md)
    def check_iteration(self):
        if self.iteration > 0:
            self.do_data_generation = False
            self.do_dft = True
            self.do_mace = True
            self.do_md = True
        return self.iteration < 10

    def initialization(self):
        """Initialize variables."""
        self.config = 0
        self.iteration = 0
        self.labelled_list = []
        self.do_data_generation = self.inputs.do_data_generation
        self.do_dft = self.inputs.do_dft
        self.do_mace = self.inputs.do_mace
        self.do_md = self.inputs.do_md



    def data_generation(self):
        """Generate data for the dataset."""
        
        inputs = self.exposed_inputs(DatasetGeneratorWorkChain, namespace="datagen")
        inputs['structures'] = self.inputs.structures

        future = self.submit(DatasetGeneratorWorkChain, **inputs)
        self.report(f'launched lammps calculation <{future.pk}>')
        self.to_context(datagen = future)
    
    def run_dft(self):
        """Run DFT calculations."""

        self.config += 1
        self.iteration += 1
        if self.iteration > 1:
            ase_list = dataset_list_to_ase_list(self.cometee_evaluation_list)
        else:
            if self.do_data_generation:
                ase_list = dataset_list_to_ase_list(self.ctx.datagen.outputs.structure_lists.global_structure_list)            
            else:
                ase_list = dataset_list_to_ase_list(self.inputs.non_labelled_list)


        for _, structure in enumerate(ase_list):
            self.config += 1
            str_data = StructureData(ase=structure)
            default_inputs = {'CONTROL': {'calculation': 'scf', 'tstress': True, 'tprnfor': True}}

            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='dft'))
            inputs.pw.structure = str_data
            inputs.metadata.call_link_label = f'dft_config_{self.config}'
            
            atm_types = list(str_data.get_symbols_set())
            pseudos = inputs.pw.pseudos
            inputs.pw.pseudos = {}
            for tp in atm_types:
                if tp in pseudos.keys():
                    inputs.pw.pseudos[tp] = pseudos[tp]
                else:
                    raise ValueError(f'Pseudopotential for {tp} not found')
            
            
            inputs.pw.parameters = Dict(recursive_merge(default_inputs, inputs.pw.parameters.get_dict()))
            
            inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

            future = self.submit(PwBaseWorkChain, **inputs)

            self.report(f'launched PwBaseWorkChain for configuration {self.config} <{future.pk}>')
            self.to_context(dft_calculations=append_(future))

    def run_mace(self):
        """Run MACE calculations."""
        inputs = self.exposed_inputs(MaceWorkChain, namespace="mace")

        if self.do_dft:
            inputs['dataset_list'] = List(self.labelled_list)
        else:
            inputs['dataset_list'] = self.inputs.labelled_list
        future = self.submit(MaceWorkChain, **inputs)
        self.to_context(mace_wc = future)
        pass

    def run_md(self):
        """Run MD calculations."""
        if self.do_mace:
            potential = self.ctx.mace_wc.outputs.mace.mace_0.aiida_model_lammps
        else:
            potential = self.inputs.mace_lammps_potential

        for _, structure in self.inputs.structures.items():
            for temp in self.inputs.md.temperatures:
                for press in self.inputs.md.pressures:
                    inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
                    inputs.structure = structure
                    inputs.temperature = Float(temp)
                    inputs.pressure = Float(press)
                    inputs.potential = potential
                    future = self.submit(LammpsWorkChain, **inputs)
                    self.to_context(md_wc=append_(future))
        # inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
        # inputs.structure = self.inputs.structures[0]
        # pass
    
    def run_md_frame_extraction(self):
        """Run MD frame extraction."""
        # for _, trajectory in self.trajectories.items():

        lammps_extracted_list = LammpsFrameExtraction(self.inputs.frame_extraction.correlation_time,
                                Int(100),
                                **self.trajectories)['lammps_extracted_list']
        self.lammps_extracted_list = lammps_extracted_list
        self.lammps_extracted_list = lammps_extracted_list
        self.out('md.lammps_extracted_list', lammps_extracted_list)
        # inputs = self.exposed_inputs(FrameExtractionWorkChain, namespace="frame_extraction")
        # inputs.trajectories = self.trajectories
        # inputs.input_structure = self.inputs.structures['s0']
        # inputs.dt = self.inputs.md.dt
        # inputs.saving_frequency = Int(100)
        # future = self.submit(FrameExtractionWorkChain, **inputs)
        # self.to_context(frame_extraction_wc=append_(future))

    def run_cometee_evaluation(self):
        inputs = self.exposed_inputs(EvaluationCalculation, namespace="cometee_evaluation")
        inputs['mace_potentials'] = self.potentials
        inputs['datasetlist'] = self.lammps_extracted_list

        future = self.submit(EvaluationCalculation, **inputs)
        self.to_context(cometee_evalutation = future)

        





    def finalize_data_generation(self):
        """Finalize."""

        self.out_many(self.exposed_outputs(self.ctx.datagen, DatasetGeneratorWorkChain, namespace="datagen"))

    def finalize_dft(self):
        dft_data = {}
        for ii, calc in enumerate(self.ctx.dft_calculations):
            if calc.exit_status == 0:
                dft_data[f'dft_{ii}'] = {
                    'output_parameters': calc.outputs.output_parameters,
                    'output_trajectory': calc.outputs.output_trajectory
                    }
        if self.do_data_generation:
            labelled_list = WriteLabelledList(non_labelled_structures = self.ctx.datagen.outputs.structure_lists.global_structure_list, **dft_data)
        else:
            labelled_list = WriteLabelledList(non_labelled_structures = self.inputs.non_labelled_list, **dft_data)
        
        self.labelled_list += labelled_list.get_list()
        self.out('dft.labelled_list', labelled_list)

    def finalize_mace(self):
        self.potentials = {}
        for key, val in self.ctx.mace_wc.outputs.mace.items():
            for k, v in val.items():
                if k == 'aiida_swa_compiled_model':
                    self.potentials[key] = v
        
            

        self.out('mace', self.ctx.mace_wc.outputs.mace)
        # self.out_many(self.exposed_outputs(self.ctx.mace_wc, MaceWorkChain, namespace="mace"))

    def finalize_md(self):
        
        md_out = {}
        self.trajectories = {}
        for ii, calc in enumerate(self.ctx.md_wc):
            # self.report(f'ii : {ii}')
            # self.report(f'calc.outputs : {calc.outputs}')
                # self.report(f'k : {k}')
                # self.report(f'calc.outputs[k] : {calc.outputs[k]}')
                for el in calc.outputs['lmp_out']:
                    # self.report(f'el : {el}')
                    # self.report(f'calc.outputs[k][el] : {calc.outputs[k][el]}')
                    if el == 'coord_atom':
                        self.trajectories[f'md_{ii}'] = calc.outputs['lmp_out'][el]
                    md_out[f'md_{ii}']={el:calc.outputs['lmp_out'][el] for el in calc.outputs['lmp_out']}
            # for out in calc.outputs:
            #     md_out[f'md_{ii}'][out] = calc.outputs[out]

        self.out('md', md_out)


        # self.out('md', self.ctx.md_wc.lmp_out)
    
    def finalize_cometee_evaluation(self):
        calc = self.ctx.cometee_evalutation
        self.cometee_evaluation_list = calc.outputs.evaluated_list
        self.out('cometee_evaluation_list', calc.outputs.evaluated_list)
        # self.out_many(self.exposed_outputs(self.ctx.cometee_evalutation, EvaluationCalculation, namespace="cometee_evaluation"))
    # def finalize_md_frame_extraction(self):
        # self.out('frame_extraction', self.ctx.frame_extraction_wc[0].outputs)
        