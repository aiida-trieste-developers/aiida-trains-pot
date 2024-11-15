# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction, if_, while_, ExitCode
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, List, Float, SinglefileData, Bool, Dict
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.common import AttributeDict
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from aiida_lammps.data.potential import LammpsPotentialData
from aiida.plugins import DataFactory
import tempfile
from ase.io.lammpsrun import read_lammps_dump_text
from io import StringIO
import numpy as np
from pathlib import Path
import os
import io
load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')
DatasetAugmentationWorkChain    = WorkflowFactory('trains_pot.datasetaugmentation')
TrainingWorkChain               = WorkflowFactory('trains_pot.training')
AbInitioLabellingWorkChain      = WorkflowFactory('trains_pot.labelling')  
LammpsWorkChain                 = WorkflowFactory('lammps.base')
EvaluationCalculation           = CalculationFactory('trains_pot.evaluation')
PESData                         = DataFactory('pesdata')

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

            extracted_frames.append({'cell': cell.tolist(),
                    'symbols': list(step_data[5]['element']),
                    'positions': [[step_data[5]['x'][jj],step_data[5]['y'][jj],step_data[5]['z'][jj]] for jj, _ in enumerate(step_data[5]['y'])],
                    'input_structure_uuid': str(input_structure_node.uuid),
                    # 'md_exploration_forces': List(list(trajectory_frames[i].get_forces())),
                    'gen_method': str('LAMMPS')
                    })
            extracted_frames[-1]['style'] = params['md']['integration']['style']
            extracted_frames[-1]['temp'] = params['md']['integration']['constraints']['temp']
            extracted_frames[-1]['timestep'] = params['control']['timestep']
            extracted_frames[-1]['id_lammps'] = lammps_id

            i = i + int(sampling_time/params['control']['timestep']/saving_frequency)

    pes_extracted_frames = PESData()    
    pes_extracted_frames.set_list(extracted_frames)  
    return {'lammps_extracted_list': pes_extracted_frames}

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

    pes_selected_list = PESData()    
    pes_selected_list.set_list(selected_list)
    return {'selected_list':pes_selected_list, 'min_energy_deviation':Float(min(energy_deviation)), 'max_energy_deviation':Float(max(energy_deviation)), 'min_forces_deviation':Float(min(forces_deviation)), 'max_forces_deviation':Float(max(forces_deviation)), 'min_stress_deviation':Float(min(stress_deviation)), 'max_stress_deviation':Float(max(stress_deviation))}

class TrainsPotWorkChain(WorkChain):
    """WorkChain to launch LAMMPS calculations."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input('do_data_set_augmentation', valid_type=Bool, default=lambda: Bool(True), help='Do data generation', required=False)
        spec.input('do_ab_initio_labelling', valid_type=Bool, default=lambda: Bool(True), help='Do ab_initio_labelling calculations', required=False)
        spec.input('do_training', valid_type=Bool, default=lambda: Bool(True), help='Do MACE calculations', required=False)
        spec.input('do_md_exploration', valid_type=Bool, default=lambda: Bool(True), help='Do md_exploration calculations', required=False)
        spec.input('max_loops', valid_type=Int, default=lambda: Int(10), help='Maximum number of active learning workflow loops', required=False)

        spec.input_namespace('lammps_input_structures', valid_type=StructureData, help='Input structures for lammps, if not specified input structures are used', required=False)
        spec.input('non_labelled_list', valid_type=PESData, help='List of non labelled structures', required=False)
        spec.input('labelled_list', valid_type=PESData, help='List of labelled structures', required=False)
        spec.input('mace_workchain_pk', valid_type=Str, help='MACE workchain pk', required=False)
        spec.input_namespace('training_lammps_potentials', valid_type=SinglefileData, help='MACE potential for md_exploration', required=False)
        spec.input_namespace('training_ase_potentials', valid_type=SinglefileData, help='MACE potential for Evaluation', required=False)        
        
        spec.input('md_exploration.lammps_params_list', valid_type=List, help='List of parameters for md_exploration', required=False)
        spec.input('md_exploration.parameters', valid_type=Dict, help='List of parameters for md_exploration', required=False)
        #spec.input('md_exploration.temperatures', valid_type=List, help='List of temperatures for md_exploration', required=False)
        #spec.input('md_exploration.pressures', valid_type=List, help='List of pressures for md_exploration', required=False)
        spec.input('potential', valid_type=SinglefileData, help='MACE potential for md_exploration', required=False)

        spec.input('frame_extraction.sampling_time', valid_type=Float, help='Correlation time for frame extraction', required=False)
        spec.input('frame_extraction.thermalization_time', valid_type=Float, default=lambda : Float(0.0), help='Thermalization time for md_exploration', required=False)

        spec.input('thr_energy', valid_type=Float, help='Threshold for energy', required=True)
        spec.input('thr_forces', valid_type=Float, help='Threshold for forces', required=True)
        spec.input('thr_stress', valid_type=Float, help='Threshold for stress', required=True)

        spec.expose_inputs(DatasetAugmentationWorkChain, namespace="data_set_augmentation", exclude=('structures'))    
        spec.expose_inputs(AbInitioLabellingWorkChain, namespace="ab_initio_labelling",  exclude=('non_labelled_list'), namespace_options={'validator': None})    
        spec.expose_inputs(TrainingWorkChain, namespace="training", exclude=('dataset'), namespace_options={'validator': None})
        spec.expose_inputs(LammpsWorkChain, namespace="md_exploration", exclude=('lammps.structure', 'lammps.potential','lammps.parameters'), namespace_options={'validator': None})
        spec.expose_inputs(EvaluationCalculation, namespace="committee_evaluation", exclude=('mace_potentials', 'datasetlist'))
        # spec.expose_inputs(FrameExtractionWorkChain, namespace="frame_extraction", exclude=('trajectories', 'input_structure', 'dt', 'saving_frequency'))

        spec.input_namespace("structures", valid_type=StructureData, required=True)
        spec.output("ab_initio_labelling.labelled_list", valid_type=PESData, help="List of configurations labelled via ab_initio_labelling")
        spec.output("md_exploration.lammps_extracted_list", valid_type=PESData, help="List of extracted frames from md_exploration trajectories")
        spec.output("committee_evaluation_list", valid_type=PESData, help="List of committee evaluated configurations")
        spec.output_namespace("md_exploration", dynamic=True, help="md_exploration outputs")        
        spec.expose_outputs(DatasetAugmentationWorkChain, namespace="data_set_augmentation") 
        spec.expose_outputs(TrainingWorkChain, namespace="training") 
        spec.exit_code(300, "LESS_THAN_2_POTENTIALS", message="Calculation did not produce more tha 1 expected potentials.",)
        # spec.expose_outputs(EvaluationCalculation, namespace="committee_evaluation")
        # spec.expose_outputs(MaceWorkChain, namespace="mace")

        
        spec.outline(
            cls.initialization,
            if_(cls.do_data_set_augmentation)(
                cls.data_set_augmentation,
                cls.finalize_data_set_augmentation),
            while_(cls.check_iteration)(
                if_(cls.do_ab_initio_labelling)(
                    cls.ab_initio_labelling,
                    cls.finalize_ab_initio_labelling),
                if_(cls.do_training)(
                    cls.training,
                    cls.finalize_training),
                if_(cls.do_md_exploration)(
                    cls.md_exploration,
                    cls.finalize_md_exploration,
                    cls.md_exploration_frame_extraction,
                    cls.run_committee_evaluation,
                    cls.finalize_committee_evaluation),
            )
            # cls.finalize,
            # cls.save_files
        )

    # @classmethod
    # def get_builder_from_protocol(cls, structures, qe_code, qe_protocol=None, qe_options=None, qe_overrides=None, **kwargs):
    #     """Return a builder"""
        
    #     builder = cls.get_builder()
    #     builder.structures = {f's{ii}':s for ii, s in enumerate(structures)}
    #     builder.ab_initio_labelling = PwBaseWorkChain.get_builder_from_protocol(*(qe_code, structures[0], qe_protocol), overrides = qe_overrides, options=qe_options, **kwargs)

    #     return builder
    
    def do_data_set_augmentation(self): return bool(self.ctx.do_data_set_augmentation)
    def do_ab_initio_labelling(self): return bool(self.ctx.do_ab_initio_labelling)
    def do_training(self): return bool(self.ctx.do_training)
    def do_md_exploration(self): return bool(self.ctx.do_md_exploration)
    def check_iteration(self):
        if self.ctx.iteration > 0:
            self.ctx.do_data_set_augmentation = False
            self.ctx.do_ab_initio_labelling = True
            self.ctx.do_training = True
            self.ctx.do_md_exploration = True
        self.ctx.iteration += 1
        return self.ctx.iteration < self.inputs.max_loops+1

    def initialization(self):
        """Initialize variables."""        
        self.ctx.iteration = 0
        if 'labelled_list' in self.inputs:
            self.ctx.labelled_list = self.inputs.labelled_list
        else:
            self.ctx.labelled_list = []
        self.ctx.do_data_set_augmentation = self.inputs.do_data_set_augmentation
        self.ctx.do_ab_initio_labelling = self.inputs.do_ab_initio_labelling
        self.ctx.do_training = self.inputs.do_training
        self.ctx.do_md_exploration = self.inputs.do_md_exploration
        self.ctx.checkpoints = []
        if not self.ctx.do_ab_initio_labelling:
            self.ctx.labelled_list = self.inputs.labelled_list

        if not self.ctx.do_training:
            self.ctx.potentials_lammps = []
            self.ctx.potentials = []
            for _, pot in self.inputs.training_lammps_potentials.items():
                self.ctx.potentials_lammps.append(pot)
            for _, pot in self.inputs.training_ase_potentials.items():
                self.ctx.potentials.append(pot)
        if 'lammps_input_structures' in self.inputs:
            self.ctx.lammps_input_structures = self.inputs.lammps_input_structures
        else:
            self.ctx.lammps_input_structures = self.inputs.structures
            


    def data_set_augmentation(self):
        """Generate data for the dataset."""
        
        inputs = self.exposed_inputs(DatasetAugmentationWorkChain, namespace="data_set_augmentation")
        inputs['structures'] = self.inputs.structures

        future = self.submit(DatasetAugmentationWorkChain, **inputs)
        self.report(f'launched lammps calculation <{future.pk}>')
        self.to_context(data_set_augmentation = future)
    
    def ab_initio_labelling(self):
        """Run ab_initio_labelling calculations."""
                
        if self.ctx.iteration > 1:
            ase_list = self.ctx.committee_evaluation_list
        else:
            if self.ctx.do_data_set_augmentation:
                ase_list = self.ctx.data_set_augmentation.outputs.structure_lists.global_structure_list       
            else:
                ase_list = self.inputs.non_labelled_list

        # Set up the inputs for LoopingLabellingWorkChain
        inputs = self.exposed_inputs(AbInitioLabellingWorkChain, namespace="ab_initio_labelling")
        inputs.non_labelled_list = ase_list
                

        # Submit LoopingLabellingWorkChain
        future = self.submit(AbInitioLabellingWorkChain, **inputs)

        self.report(f'Launched AbInitioLabellingWorkChain with ase_list <{future.pk}>')
        self.to_context(ab_initio_labelling = future)

    def training(self):
        """Run training calculations."""
        dataset = PESData()
        if self.ctx.do_ab_initio_labelling:            
            dataset.set_list(self.ctx.labelled_list)
        else:
            dataset = self.inputs.labelled_list   
        inputs = self.exposed_inputs(TrainingWorkChain, namespace="training")
        inputs.dataset = dataset
        if self.ctx.iteration > 1:
            inputs['checkpoints'] = {f"chkpt_{ii+1}": self.ctx.checkpoints[-ii] for ii in range(min(len(self.ctx.checkpoints), self.inputs.training.num_potentials.value))}
      
      
        future = self.submit(TrainingWorkChain, **inputs)

        self.report(f'Launched TrainingWorkChain with dataset_list <{future.pk}>')
        self.to_context(training = future)            

    def md_exploration(self):
        """Run md_exploration."""
        potential = self.ctx.potentials_lammps[-1]        
        for _, structure in self.ctx.lammps_input_structures.items():
            inputs = self.exposed_inputs(LammpsWorkChain, namespace="md_exploration")
            inputs.lammps.structure = structure
            inputs.lammps.potential = generate_potential(potential)
            params_list=list(self.inputs.md_exploration.lammps_params_list)
            parameters=AttributeDict(self.inputs.md_exploration.parameters)
            parameters.dump.dump_rate = int(self.inputs.frame_extraction.sampling_time/parameters.control.timestep)
            for params_md_exploration in params_list:            
                parameters.md = dict(params_md_exploration)            
                inputs.lammps.parameters = Dict(parameters)                
                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_exploration_wc=append_(future))
            #inputs.temperature = Float(temp)
            #inputs.pressure = Float(press)
            #for temp in self.inputs.md_exploration.temperatures:
            #    for press in self.inputs.md_exploration.pressures:
            #        inputs = self.exposed_inputs(LammpsWorkChain, namespace="md_exploration")
            #        inputs.structure = structure
                    #inputs.temperature = Float(temp)
                    #inputs.pressure = Float(press)
            #        inputs.potential = potential
            #        future = self.submit(LammpsWorkChain, **inputs)
            #        self.to_context(md_exploration_wc=append_(future))
        # inputs = self.exposed_inputs(LammpsWorkChain, namespace="md_exploration")
        # inputs.structure = self.inputs.structures[0]
        # pass
    
    def md_exploration_frame_extraction(self):
        """Run md_exploration frame extraction."""
        # for _, trajectory in self.ctx.trajectories.items():        
        parameters=AttributeDict(self.inputs.md_exploration.parameters)
        dump_rate = int(self.inputs.frame_extraction.sampling_time/parameters.control.timestep)
        lammps_extracted_list = LammpsFrameExtraction(self.inputs.frame_extraction.sampling_time,
                                dump_rate,
                                thermalization_time = self.inputs.frame_extraction.thermalization_time, 
                                **self.ctx.trajectories)['lammps_extracted_list']
        self.ctx.lammps_extracted_list = lammps_extracted_list
        self.out('md_exploration.lammps_extracted_list', lammps_extracted_list)
        # inputs = self.exposed_inputs(FrameExtractionWorkChain, namespace="frame_extraction")
        # inputs.trajectories = self.ctx.trajectories
        # inputs.input_structure = self.inputs.structures['s0']
        # inputs.dt = self.inputs.md_exploration.dt
        # inputs.saving_frequency = Int(100)
        # future = self.submit(FrameExtractionWorkChain, **inputs)
        # self.to_context(frame_extraction_wc=append_(future))

    def run_committee_evaluation(self):
        inputs = self.exposed_inputs(EvaluationCalculation, namespace="committee_evaluation")
        inputs['mace_potentials'] = {f"pot_{ii}": self.ctx.potentials[ii] for ii in range(len(self.ctx.potentials))}
        inputs['datasetlist'] = self.ctx.lammps_extracted_list

        future = self.submit(EvaluationCalculation, **inputs)
        self.to_context(committee_evalutation = future)  

    def finalize_data_set_augmentation(self):
        """Finalize."""

        self.out_many(self.exposed_outputs(self.ctx.data_set_augmentation, DatasetAugmentationWorkChain, namespace="data_set_augmentation"))

    def finalize_ab_initio_labelling(self):
        self.ctx.labelled_list += self.ctx.ab_initio_labelling.outputs.ab_initio_labelling_data
        self.out('ab_initio_labelling.labelled_list', self.ctx.ab_initio_labelling.outputs.ab_initio_labelling_data)
        self.ctx.ab_initio_labelling_calculations = []

    def finalize_training(self):
        
        if len(self.ctx.training.outputs.training) < 2:
            return self.exit_codes.LESS_THAN_2_POTENTIALS                     

        self.ctx.potentials = []
        self.ctx.potentials_lammps = []
        self.ctx.checkpoints = []        
        for ii, calc in enumerate(self.ctx.training.outputs.training.values()):                        
            for key, value in calc.items():                
                if key == 'swa_ase_model':
                    self.ctx.potentials.append(value)                   
                elif key == 'checkpoints':
                    self.ctx.checkpoints.append(value)                    
                elif key == 'swa_model_lammps':
                    self.ctx.potentials_lammps.append(value)                   
                       
        self.ctx.labelled_list = self.ctx.training.outputs.global_splitted        
    def finalize_md_exploration(self):

        md_exploration_out = {}
        self.ctx.trajectories = {}
        calc_no_exception = False
        for ii, calc in enumerate(self.ctx.md_exploration_wc):
            self.report(f'md_exploration_{ii} exit status: {calc.exit_status}')
            if calc.exit_status == 0:
                calc_no_exception = True
                self.report(f'md_exploration_{ii} exit0')
            # self.report(f'ii : {ii}')
            # self.report(f'calc.outputs : {calc.outputs}')
                # self.report(f'k : {k}')
                # self.report(f'calc.outputs[k] : {calc.outputs[k]}')                
                for el in calc.outputs:
                    # self.report(f'el : {el}')
                    # self.report(f'calc.outputs[k][el] : {calc.outputs[k][el]}')
                    if el == 'trajectories':
                        self.ctx.trajectories[f'md_exploration_{ii}'] = calc.outputs[el]
                    md_exploration_out[f'md_exploration_{ii}']={el:calc.outputs[el] for el in calc.outputs}
            # for out in calc.outputs:
            #     md_exploration_out[f'md_exploration_{ii}'][out] = calc.outputs[out]
        self.ctx.md_exploration_wc = []
        self.out('md_exploration', md_exploration_out)
        if not calc_no_exception:
            return ExitCode(309, 'No md_exploration calculation ended correctly')

        # self.out('md_exploration', self.ctx.md_exploration_wc.lmp_out)
    
    def finalize_committee_evaluation(self):
        calc = self.ctx.committee_evalutation

        selected = SelectToLabel(calc.outputs.evaluated_list, self.inputs.thr_energy, self.inputs.thr_forces, self.inputs.thr_stress)
        self.ctx.committee_evaluation_list = selected['selected_list']
        self.report(f'Structures selected for labelling: {len(self.ctx.committee_evaluation_list.get_list())}/{len(calc.outputs.evaluated_list.get_list())}')
        self.report(f'Min energy deviation: {round(selected["min_energy_deviation"].value,2)} eV, Max energy deviation: {round(selected["max_energy_deviation"].value,2)} eV')
        self.report(f'Min forces deviation: {round(selected["min_forces_deviation"].value,2)} eV/Å, Max forces deviation: {round(selected["max_forces_deviation"].value,2)} eV/Å')
        self.report(f'Min stress deviation: {round(selected["min_stress_deviation"].value,2)} kbar, Max stress deviation: {round(selected["max_stress_deviation"].value,2)} kbar')
        self.out('committee_evaluation_list', calc.outputs.evaluated_list)
        # self.out_many(self.exposed_outputs(self.ctx.committee_evalutation, EvaluationCalculation, namespace="committee_evaluation"))
    # def finalize_md_exploration_frame_extraction(self):
        # self.out('frame_extraction', self.ctx.frame_extraction_wc[0].outputs)


