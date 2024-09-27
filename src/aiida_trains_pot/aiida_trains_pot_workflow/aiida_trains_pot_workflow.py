# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction, if_, while_, ExitCode
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, List, Float, SinglefileData, Bool, Dict
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.common import AttributeDict
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from aiida_lammps.data.potential import LammpsPotentialData
import tempfile
from ase.io.lammpsrun import read_lammps_dump_text
from io import StringIO
import numpy as np
from pathlib import Path
import os
import io
load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')
DatasetGeneratorWorkChain   = WorkflowFactory('trains_pot.datageneration')
PwBaseWorkChain             = WorkflowFactory('quantumespresso.pw.base')
MaceWorkChain               = WorkflowFactory('trains_pot.macetrain')
LammpsWorkChain             = WorkflowFactory('lammps.base')
EvaluationCalculation       = CalculationFactory('trains_pot.evaluation')

def generate_potential(potential) -> LammpsPotentialData:
        """
        Generate the potential to be used in the calculation.

        Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

        :return: potential to do the calculation
        :rtype: LammpsPotentialData
        """

        potential_parameters = {
            "species": ["C"],
            "atom_style": "atomic",        
            "units": "metal",
            "extra_tags": {},                
        }

        # Assuming you have a trained MACE model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with potential.open(mode='rb') as potential_handle:
                potential_content = potential_handle.read()
            tmp_file.write(potential_content)
            tmp_file_path = tmp_file.name
            
        potential = LammpsPotentialData.get_or_create(
            #source=binary_stream,
            source = Path(tmp_file_path),
            pair_style="mace no_domain_decomposition",
            **potential_parameters,
        )

        os.remove(tmp_file_path)
    
        return potential


def dataset_list_to_ase_list(dataset_list):
    """Convert dataset list to an ASE list."""

    ase_list = []

    for config in dataset_list:
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
def LammpsFrameExtraction(sampling_time, saving_frequency, thermalization_time=0, **trajectories):
    """Extract frames from trajectory."""


    extracted_frames = []      
    for _, trajectory in trajectories.items():

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
            cell = step_data.cell

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

            i = i + int(sampling_time/params['control']['timestep']/saving_frequency)

    return {'lammps_extracted_list': List(list=extracted_frames)}

@calcfunction
def SelectToLabel(evaluated_list, thr_energy, thr_forces, thr_stress):
    """Select configurations to label."""
    selected_list = []
    energy_deviation = []
    forces_deviation = []
    stress_deviation = []
    for config in evaluated_list.get_list():
        energy_deviation.append(config['energy_deviation'])
        forces_deviation.append(config['forces_deviation'])
        stress_deviation.append(config['stress_deviation'])
        if config['energy_deviation'] > thr_energy or config['forces_deviation'] > thr_forces or config['stress_deviation'] > thr_stress:
            selected_list.append(config)

    return {'selected_list':List(list=selected_list), 'min_energy_deviation':Float(min(energy_deviation)), 'max_energy_deviation':Float(max(energy_deviation)), 'min_forces_deviation':Float(min(forces_deviation)), 'max_forces_deviation':Float(max(forces_deviation)), 'min_stress_deviation':Float(min(stress_deviation)), 'max_stress_deviation':Float(max(stress_deviation))}

class TrainsPotWorkChain(WorkChain):
    """WorkChain to launch LAMMPS calculations."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input('do_data_generation', valid_type=Bool, default=lambda: Bool(True), help='Do data generation', required=False)
        spec.input('do_dft', valid_type=Bool, default=lambda: Bool(True), help='Do DFT calculations', required=False)
        spec.input('do_mace', valid_type=Bool, default=lambda: Bool(True), help='Do MACE calculations', required=False)
        spec.input('do_md', valid_type=Bool, default=lambda: Bool(True), help='Do MD calculations', required=False)
        spec.input('max_loops', valid_type=Int, default=lambda: Int(10), help='Maximum number of active learning workflow loops', required=False)

        spec.input_namespace('lammps_input_structures', valid_type=StructureData, help='Input structures for lammps, if not specified input structures are used', required=False)
        spec.input('non_labelled_list', valid_type=List, help='List of non labelled structures', required=False)
        spec.input('labelled_list', valid_type=List, help='List of labelled structures', required=False)
        spec.input('mace_workchain_pk', valid_type=Str, help='MACE workchain pk', required=False)
        spec.input_namespace('mace_lammps_potentials', valid_type=SinglefileData, help='MACE potential for MD', required=False)
        spec.input_namespace('mace_ase_potentials', valid_type=SinglefileData, help='MACE potential for Evaluation', required=False)

        spec.input('md.md_params_list', valid_type=List, help='List of parameters for MD', required=False)
        spec.input('md.parameters', valid_type=Dict, help='List of parameters for MD', required=False)
        #spec.input('md.temperatures', valid_type=List, help='List of temperatures for MD', required=False)
        #spec.input('md.pressures', valid_type=List, help='List of pressures for MD', required=False)
        spec.input('potential', valid_type=SinglefileData, help='MACE potential for MD', required=False)

        spec.input('frame_extraction.sampling_time', valid_type=Float, help='Correlation time for frame extraction', required=False)
        spec.input('frame_extraction.thermalization_time', valid_type=Float, default=lambda : Float(0.0), help='Thermalization time for MD', required=False)

        spec.input('thr_energy', valid_type=Float, help='Threshold for energy', required=True)
        spec.input('thr_forces', valid_type=Float, help='Threshold for forces', required=True)
        spec.input('thr_stress', valid_type=Float, help='Threshold for stress', required=True)

        spec.expose_inputs(DatasetGeneratorWorkChain, namespace="datagen", exclude=('structures'))
        spec.expose_inputs(PwBaseWorkChain, namespace="dft", exclude=('pw.structure',), namespace_options={'validator': None})
        spec.expose_inputs(MaceWorkChain, namespace="mace", exclude=('dataset_list',), namespace_options={'validator': None})
        spec.expose_inputs(LammpsWorkChain, namespace="md", exclude=('lammps.structure', 'lammps.potential','lammps.parameters'), namespace_options={'validator': None})
        spec.expose_inputs(EvaluationCalculation, namespace="committee_evaluation", exclude=('mace_potentials', 'datasetlist'))
        # spec.expose_inputs(FrameExtractionWorkChain, namespace="frame_extraction", exclude=('trajectories', 'input_structure', 'dt', 'saving_frequency'))

        spec.input_namespace("structures", valid_type=StructureData, required=True)

        spec.output("dft.labelled_list", valid_type=List, help="List of configurations labelled via DFT")
        spec.output("md.lammps_extracted_list", valid_type=List, help="List of extracted frames from MD trajectories")
        spec.output("committee_evaluation_list", valid_type=List, help="List of committee evaluated configurations")
        spec.output_namespace("md", dynamic=True, help="MD outputs")
        spec.output_namespace("mace", dynamic=True, help="MACE outputs")
        spec.expose_outputs(DatasetGeneratorWorkChain, namespace="datagen")
        # spec.expose_outputs(EvaluationCalculation, namespace="committee_evaluation")
        # spec.expose_outputs(MaceWorkChain, namespace="mace")

        
        spec.outline(
            cls.initialization,
            if_(cls.ver_do_data_generation)(
                cls.data_generation,
                cls.finalize_data_generation),
            while_(cls.check_iteration)(
                if_(cls.ver_do_dft)(
                    cls.run_dft,
                    cls.finalize_dft),
                if_(cls.ver_do_mace)(
                    cls.run_mace,
                    cls.finalize_mace),
                if_(cls.ver_do_md)(
                    cls.run_md,
                    cls.finalize_md,
                    cls.run_md_frame_extraction,
                    cls.run_committee_evaluation,
                    cls.finalize_committee_evaluation),
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
    
    def ver_do_data_generation(self): return bool(self.ctx.do_data_generation)
    def ver_do_dft(self): return bool(self.ctx.do_dft)
    def ver_do_mace(self): return bool(self.ctx.do_mace)
    def ver_do_md(self): return bool(self.ctx.do_md)
    def check_iteration(self):
        if self.ctx.iteration > 0:
            self.ctx.do_data_generation = False
            self.ctx.do_dft = True
            self.ctx.do_mace = True
            self.ctx.do_md = True
        self.ctx.iteration += 1
        return self.ctx.iteration < self.inputs.max_loops+1

    def initialization(self):
        """Initialize variables."""
        self.ctx.config = 0
        self.ctx.iteration = 0
        if 'labelled_list' in self.inputs:
            self.ctx.labelled_list = self.inputs.labelled_list.get_list()
        else:
            self.ctx.labelled_list = []
        self.ctx.do_data_generation = self.inputs.do_data_generation
        self.ctx.do_dft = self.inputs.do_dft
        self.ctx.do_mace = self.inputs.do_mace
        self.ctx.do_md = self.inputs.do_md
        self.ctx.checkpoints = []
        if not self.ctx.do_dft:
            self.ctx.labelled_list = self.inputs.labelled_list.get_list()

        if not self.ctx.do_mace:
            self.ctx.potentials_lammps = []
            self.ctx.potentials = []
            for _, pot in self.inputs.mace_lammps_potentials.items():
                self.ctx.potentials_lammps.append(pot)
            for _, pot in self.inputs.mace_ase_potentials.items():
                self.ctx.potentials.append(pot)
        if 'lammps_input_structures' in self.inputs:
            self.ctx.lammps_input_structures = self.inputs.lammps_input_structures
        else:
            self.ctx.lammps_input_structures = self.inputs.structures
            


    def data_generation(self):
        """Generate data for the dataset."""
        
        inputs = self.exposed_inputs(DatasetGeneratorWorkChain, namespace="datagen")
        inputs['structures'] = self.inputs.structures

        future = self.submit(DatasetGeneratorWorkChain, **inputs)
        self.report(f'launched lammps calculation <{future.pk}>')
        self.to_context(datagen = future)
    
    def run_dft(self):
        """Run DFT calculations."""

        self.ctx.config += 1
        
        if self.ctx.iteration > 1:
            ase_list = dataset_list_to_ase_list(self.ctx.committee_evaluation_list)
        else:
            if self.ctx.do_data_generation:
                ase_list = dataset_list_to_ase_list(self.ctx.datagen.outputs.structure_lists.global_structure_list.get_list())            
            else:
                ase_list = dataset_list_to_ase_list(self.inputs.non_labelled_list.get_list())


        for _, structure in enumerate(ase_list):
            self.ctx.config += 1
            str_data = StructureData(ase=structure)
            default_inputs = {'CONTROL': {'calculation': 'scf', 'tstress': True, 'tprnfor': True}}

            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='dft'))
            inputs.pw.structure = str_data
            inputs.metadata.call_link_label = f'dft_config_{self.ctx.config}'
            
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

            self.report(f'launched PwBaseWorkChain for configuration {self.ctx.config} <{future.pk}>')
            self.to_context(dft_calculations=append_(future))

    def run_mace(self):
        """Run MACE calculations."""
        inputs = self.exposed_inputs(MaceWorkChain, namespace="mace")

        if self.ctx.do_dft:
            inputs['dataset_list'] = List(self.ctx.labelled_list)
        else:
            inputs['dataset_list'] = self.inputs.labelled_list
        if self.ctx.iteration > 1:
            inputs['checkpoints'] = {f"chkpt_{ii+1}": self.ctx.checkpoints[-ii] for ii in range(min(len(self.ctx.checkpoints), self.inputs.mace.num_potentials.value))}
            inputs.mace['restart'] = Bool(True)
        future = self.submit(MaceWorkChain, **inputs)
        self.to_context(mace_wc = future)
        pass

    def run_md(self):
        """Run MD calculations."""
        potential = self.ctx.potentials_lammps[-1]        
        for _, structure in self.ctx.lammps_input_structures.items():
            inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
            inputs.lammps.structure = structure
            inputs.lammps.potential = generate_potential(potential)
            params_list=list(self.inputs.md.md_params_list)
            parameters=AttributeDict(self.inputs.md.parameters)
            parameters.dump.dump_rate = int(self.inputs.frame_extraction.sampling_time/parameters.control.timestep)
            for params_md in params_list:            
                parameters.md = dict(params_md)            
                inputs.lammps.parameters = Dict(parameters)                
                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_wc=append_(future))
            #inputs.temperature = Float(temp)
            #inputs.pressure = Float(press)
            #for temp in self.inputs.md.temperatures:
            #    for press in self.inputs.md.pressures:
            #        inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
            #        inputs.structure = structure
                    #inputs.temperature = Float(temp)
                    #inputs.pressure = Float(press)
            #        inputs.potential = potential
            #        future = self.submit(LammpsWorkChain, **inputs)
            #        self.to_context(md_wc=append_(future))
        # inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
        # inputs.structure = self.inputs.structures[0]
        # pass
    
    def run_md_frame_extraction(self):
        """Run MD frame extraction."""
        # for _, trajectory in self.ctx.trajectories.items():        
        parameters=AttributeDict(self.inputs.md.parameters)
        dump_rate = int(self.inputs.frame_extraction.sampling_time/parameters.control.timestep)
        lammps_extracted_list = LammpsFrameExtraction(self.inputs.frame_extraction.sampling_time,
                                dump_rate,
                                thermalization_time = self.inputs.frame_extraction.thermalization_time, 
                                **self.ctx.trajectories)['lammps_extracted_list']
        self.ctx.lammps_extracted_list = lammps_extracted_list
        self.out('md.lammps_extracted_list', lammps_extracted_list)
        # inputs = self.exposed_inputs(FrameExtractionWorkChain, namespace="frame_extraction")
        # inputs.trajectories = self.ctx.trajectories
        # inputs.input_structure = self.inputs.structures['s0']
        # inputs.dt = self.inputs.md.dt
        # inputs.saving_frequency = Int(100)
        # future = self.submit(FrameExtractionWorkChain, **inputs)
        # self.to_context(frame_extraction_wc=append_(future))

    def run_committee_evaluation(self):
        inputs = self.exposed_inputs(EvaluationCalculation, namespace="committee_evaluation")
        inputs['mace_potentials'] = {f"pot_{ii}": self.ctx.potentials[ii] for ii in range(len(self.ctx.potentials))}
        inputs['datasetlist'] = self.ctx.lammps_extracted_list

        future = self.submit(EvaluationCalculation, **inputs)
        self.to_context(committee_evalutation = future)

        





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
        if self.ctx.do_data_generation:
            labelled_list = WriteLabelledList(non_labelled_structures = self.ctx.datagen.outputs.structure_lists.global_structure_list, **dft_data)
        elif self.ctx.iteration > 1:
            labelled_list = WriteLabelledList(non_labelled_structures = self.ctx.committee_evaluation_list, **dft_data)
        else:
            labelled_list = WriteLabelledList(non_labelled_structures = self.inputs.non_labelled_list, **dft_data)
        
        self.ctx.labelled_list += labelled_list.get_list()
        self.out('dft.labelled_list', labelled_list)
        self.ctx.dft_calculations = []

    def finalize_mace(self):
        self.ctx.potentials = []
        self.ctx.potentials_lammps = []
        self.ctx.checkpoints = []
        for key, val in self.ctx.mace_wc.outputs.mace.items():
            for k, v in val.items():
                if k == 'aiida_swa_compiled_model':
                    self.ctx.potentials.append(v)
                elif k == 'checkpoints':
                    self.ctx.checkpoints.append(v)
                elif k == 'aiida_swa_model_lammps':
                    self.ctx.potentials_lammps.append(v)
            

        self.out('mace', self.ctx.mace_wc.outputs.mace)
        self.ctx.labelled_list = self.ctx.mace_wc.outputs.global_list_splitted.get_list()
        # self.out_many(self.exposed_outputs(self.ctx.mace_wc, MaceWorkChain, namespace="mace"))

    def finalize_md(self):

        md_out = {}
        self.ctx.trajectories = {}
        calc_no_exception = False
        for ii, calc in enumerate(self.ctx.md_wc):
            self.report(f'md_{ii} exit status: {calc.exit_status}')
            if calc.exit_status == 0:
                calc_no_exception = True
                self.report(f'md_{ii} exit0')
            # self.report(f'ii : {ii}')
            # self.report(f'calc.outputs : {calc.outputs}')
                # self.report(f'k : {k}')
                # self.report(f'calc.outputs[k] : {calc.outputs[k]}')                
                for el in calc.outputs:
                    # self.report(f'el : {el}')
                    # self.report(f'calc.outputs[k][el] : {calc.outputs[k][el]}')
                    if el == 'trajectories':
                        self.ctx.trajectories[f'md_{ii}'] = calc.outputs[el]
                    md_out[f'md_{ii}']={el:calc.outputs[el] for el in calc.outputs}
            # for out in calc.outputs:
            #     md_out[f'md_{ii}'][out] = calc.outputs[out]
        self.ctx.md_wc = []
        self.out('md', md_out)
        if not calc_no_exception:
            return ExitCode(309, 'No MD calculation ended correctly')

        # self.out('md', self.ctx.md_wc.lmp_out)
    
    def finalize_committee_evaluation(self):
        calc = self.ctx.committee_evalutation

        selected = SelectToLabel(calc.outputs.evaluated_list, self.inputs.thr_energy, self.inputs.thr_forces, self.inputs.thr_stress)
        self.ctx.committee_evaluation_list = selected['selected_list'].get_list()
        self.report(f'Structures selected for labelling: {len(self.ctx.committee_evaluation_list)}/{len(calc.outputs.evaluated_list.get_list())}')
        self.report(f'Min energy deviation: {round(selected["min_energy_deviation"].value,2)} eV, Max energy deviation: {round(selected["max_energy_deviation"].value,2)} eV')
        self.report(f'Min forces deviation: {round(selected["min_forces_deviation"].value,2)} eV/Å, Max forces deviation: {round(selected["max_forces_deviation"].value,2)} eV/Å')
        self.report(f'Min stress deviation: {round(selected["min_stress_deviation"].value,2)} kbar, Max stress deviation: {round(selected["max_stress_deviation"].value,2)} kbar')
        self.out('committee_evaluation_list', calc.outputs.evaluated_list)
        # self.out_many(self.exposed_outputs(self.ctx.committee_evalutation, EvaluationCalculation, namespace="committee_evaluation"))
    # def finalize_md_frame_extraction(self):
        # self.out('frame_extraction', self.ctx.frame_extraction_wc[0].outputs)


