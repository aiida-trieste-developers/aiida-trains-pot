# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction, if_, while_, ExitCode
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, List, Float, SinglefileData, Bool, Dict, load_node
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.common import AttributeDict
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from aiida.plugins import DataFactory
from ase.io.lammpsrun import read_lammps_dump_text
from io import StringIO
import numpy as np
import io
load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')
DatasetAugmentationWorkChain    = WorkflowFactory('trains_pot.datasetaugmentation')
TrainingWorkChain               = WorkflowFactory('trains_pot.training')
AbInitioLabellingWorkChain      = WorkflowFactory('trains_pot.labelling')  
MDExplorationWorkChain          = WorkflowFactory('trains_pot.mdexploration')
EvaluationCalculation           = CalculationFactory('trains_pot.evaluation')
PESData                         = DataFactory('pesdata')




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
        spec.input('do_dataset_augmentation', valid_type=Bool, default=lambda: Bool(True), help='Do data generation', required=False)
        spec.input('do_ab_initio_labelling', valid_type=Bool, default=lambda: Bool(True), help='Do ab_initio_labelling calculations', required=False)
        spec.input('do_training', valid_type=Bool, default=lambda: Bool(True), help='Do MACE calculations', required=False)
        spec.input('do_md_exploration', valid_type=Bool, default=lambda: Bool(True), help='Do md_exploration calculations', required=False)
        spec.input('max_loops', valid_type=Int, default=lambda: Int(10), help='Maximum number of active learning workflow loops', required=False)

        spec.input_namespace('lammps_input_structures', valid_type=StructureData, help='Input structures for lammps, if not specified input structures are used', required=False)
        spec.input('non_labelled_list', valid_type=PESData, help='List of non labelled structures', required=False)
        spec.input('labelled_list', valid_type=PESData, help='List of labelled structures', required=False)

        spec.input_namespace('models_lammps', valid_type=SinglefileData, help='MACE potential for md_exploration', required=False)
        spec.input_namespace('models_ase', valid_type=SinglefileData, help='MACE potential for Evaluation', required=False) 
        spec.input('md_exploration.parameters', valid_type=Dict, help='List of parameters for md_exploration', required=False)        
        
        spec.input('potential', valid_type=SinglefileData, help='MACE potential for md_exploration', required=False)

        spec.input('frame_extraction.sampling_time', valid_type=Float, help='Correlation time for frame extraction', required=False)
        spec.input('frame_extraction.thermalization_time', valid_type=Float, default=lambda : Float(0.0), help='Thermalization time for md_exploration', required=False)

        spec.input('thr_energy', valid_type=Float, help='Threshold for energy', required=True)
        spec.input('thr_forces', valid_type=Float, help='Threshold for forces', required=True)
        spec.input('thr_stress', valid_type=Float, help='Threshold for stress', required=True)

        spec.expose_inputs(DatasetAugmentationWorkChain, namespace="dataset_augmentation", exclude=('structures'))    
        spec.expose_inputs(AbInitioLabellingWorkChain, namespace="ab_initio_labelling",  exclude=('non_labelled_list'), namespace_options={'validator': None})    
        spec.expose_inputs(TrainingWorkChain, namespace="training", exclude=('dataset'), namespace_options={'validator': None})
        spec.expose_inputs(MDExplorationWorkChain, namespace="md_exploration", exclude=('potential_lammps', 'lammps_input_structures','sampling_time'), namespace_options={'validator': None})
        spec.expose_inputs(EvaluationCalculation, namespace="committee_evaluation", exclude=('mace_potentials', 'datasetlist'))        
        spec.input_namespace("structures", valid_type=StructureData, required=True)
        spec.output("ab_initio_labelling.labelled_list", valid_type=PESData, help="List of configurations labelled via ab_initio_labelling")
        spec.output("md_exploration.lammps_extracted_list", valid_type=PESData, help="List of extracted frames from md_exploration trajectories")
        spec.output("committee_evaluation_list", valid_type=PESData, help="List of committee evaluated configurations")                
        spec.expose_outputs(DatasetAugmentationWorkChain, namespace="dataset_augmentation") 
        #spec.expose_outputs(TrainingWorkChain, namespace="training") 
        #spec.expose_outputs(MDExplorationWorkChain, namespace="md_exploration") 
        spec.exit_code(309, "LESS_THAN_2_POTENTIALS", message="Calculation didn't produce more tha 1 expected potentials.",)
        spec.exit_code(309, "NO_MD_CALCULATIONS", message="Calculation didn't produce any MD calculations.",)
        

        
        spec.outline(
            cls.initialization,
            if_(cls.do_dataset_augmentation)(
                cls.dataset_augmentation,
                cls.finalize_dataset_augmentation),
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
        )
    
    def do_dataset_augmentation(self): return bool(self.ctx.do_dataset_augmentation)
    def do_ab_initio_labelling(self): return bool(self.ctx.do_ab_initio_labelling)
    def do_training(self): return bool(self.ctx.do_training)
    def do_md_exploration(self): return bool(self.ctx.do_md_exploration)
    def check_iteration(self):
        if self.ctx.iteration > 0:
            self.ctx.do_dataset_augmentation = False
            self.ctx.do_ab_initio_labelling = True
            self.ctx.do_training = True
            self.ctx.do_md_exploration = True
        self.ctx.iteration += 1
        return self.ctx.iteration < self.inputs.max_loops+1

    def initialization(self):
        """Initialize variables."""        
        self.ctx.iteration = 0
        if 'labelled_list' in self.inputs:
            self.ctx.labelled_list = self.inputs.labelled_list.get_list()
        else:
            self.ctx.labelled_list = []
        self.ctx.do_dataset_augmentation = self.inputs.do_dataset_augmentation
        self.ctx.do_ab_initio_labelling = self.inputs.do_ab_initio_labelling
        self.ctx.do_training = self.inputs.do_training
        self.ctx.do_md_exploration = self.inputs.do_md_exploration
        self.ctx.potential_checkpoints = []
        if not self.ctx.do_ab_initio_labelling:
            self.ctx.labelled_list = self.inputs.labelled_list.get_list()

        if not self.ctx.do_training:
            self.ctx.potentials_lammps = []
            self.ctx.potentials_ase = []
            for _, pot in self.inputs.models_lammps.items():
                self.ctx.potentials_lammps.append(pot)
            for _, pot in self.inputs.models_ase.items():
                self.ctx.potentials_ase.append(pot)
        if 'lammps_input_structures' in self.inputs:
            self.ctx.lammps_input_structures = self.inputs.lammps_input_structures
        else:
            self.ctx.lammps_input_structures = self.inputs.structures
            


    def dataset_augmentation(self):
        """Generate data for the dataset."""
        
        inputs = self.exposed_inputs(DatasetAugmentationWorkChain, namespace="dataset_augmentation")
        inputs['structures'] = self.inputs.structures

        future = self.submit(DatasetAugmentationWorkChain, **inputs)
        self.report(f'launched lammps calculation <{future.pk}>')
        self.to_context(dataset_augmentation = future)
    
    def ab_initio_labelling(self):
        """Run ab_initio_labelling calculations."""
                
        if self.ctx.iteration > 1:
            ase_list = self.ctx.committee_evaluation_list
        else:
            if self.ctx.do_dataset_augmentation:
                ase_list = self.ctx.dataset_augmentation.outputs.structure_lists.global_structure_list       
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
            inputs['checkpoints'] = {f"chkpt_{ii+1}": self.ctx.potential_checkpoints[-ii] for ii in range(min(len(self.ctx.potential_checkpoints), self.inputs.training.num_potentials.value))}
      
      
        future = self.submit(TrainingWorkChain, **inputs)

        self.report(f'Launched TrainingWorkChain with dataset_list <{future.pk}>')
        self.to_context(training = future)            

    def md_exploration(self):
        """Run md_exploration."""
        inputs = self.exposed_inputs(MDExplorationWorkChain, namespace="md_exploration")
        inputs.potential_lammps = self.ctx.potentials_lammps[-1]
        inputs.lammps_input_structures = self.ctx.lammps_input_structures
        inputs.sampling_time = self.inputs.frame_extraction.sampling_time

        future = self.submit(MDExplorationWorkChain, **inputs)

        self.report(f'Launched MDExplorationWorkChain with dataset_list <{future.pk}>')
        self.to_context(md_exploration = future)
    
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

    def run_committee_evaluation(self):
        inputs = self.exposed_inputs(EvaluationCalculation, namespace="committee_evaluation")
        inputs['mace_potentials'] = {f"pot_{ii}": self.ctx.potentials_ase[ii] for ii in range(len(self.ctx.potentials_ase))}
        inputs['datasetlist'] = self.ctx.lammps_extracted_list

        future = self.submit(EvaluationCalculation, **inputs)
        self.to_context(committee_evaluation = future)  

    def finalize_dataset_augmentation(self):
        """Finalize."""

        self.out_many(self.exposed_outputs(self.ctx.dataset_augmentation, DatasetAugmentationWorkChain, namespace="dataset_augmentation"))

    def finalize_ab_initio_labelling(self):
        self.ctx.labelled_list += self.ctx.ab_initio_labelling.outputs.ab_initio_labelling_data.get_list()
        self.out('ab_initio_labelling.labelled_list', self.ctx.ab_initio_labelling.outputs.ab_initio_labelling_data)
        self.ctx.ab_initio_labelling_calculations = []

    def finalize_training(self):
        
        if len(self.ctx.training.outputs.training) < 2:
            return self.exit_codes.LESS_THAN_2_POTENTIALS                     

        self.ctx.potentials_ase = []
        self.ctx.potentials_lammps = []
        self.ctx.potential_checkpoints = []        
        for ii, calc in enumerate(self.ctx.training.outputs.training.values()):
            if "checkpoints" in calc:
                self.ctx.potential_checkpoints.append(calc['checkpoints'])
            if "model_stage2_ase" in calc:
                self.ctx.potentials_ase.append(calc['model_stage2_ase'])
            elif "model_stage1_ase" in calc:
                self.ctx.potentials_ase.append(calc['model_stage1_ase'])
            if "model_stage2_lammps" in calc:
                self.ctx.potentials_lammps.append(calc['model_stage2_lammps'])
            elif "model_stage1_lammps" in calc:
                self.ctx.potentials_lammps.append(calc['model_stage1_lammps'])                  
                       
        self.ctx.labelled_list = self.ctx.training.outputs.global_splitted.get_list()       


    def finalize_md_exploration(self):

        if len(self.ctx.md_exploration.outputs.md_exploration) < 1:
            return self.exit_codes.NO_MD_CALCULATIONS
        
        self.ctx.trajectories = {}        
        for ii, calc in enumerate(self.ctx.md_exploration.outputs.md_exploration.values()):            
            
            for key, value in calc.items():
                if key == 'trajectories':
                    self.ctx.trajectories[f'md_exploration_{ii}'] = value
        self.ctx.md_exploration = []          
    
    def finalize_committee_evaluation(self):
        calc = self.ctx.committee_evaluation

        selected = SelectToLabel(calc.outputs.evaluated_list, self.inputs.thr_energy, self.inputs.thr_forces, self.inputs.thr_stress)
        self.ctx.committee_evaluation_list = selected['selected_list']
        self.report(f'Structures selected for labelling: {len(self.ctx.committee_evaluation_list.get_list())}/{len(calc.outputs.evaluated_list.get_list())}')
        self.report(f'Min energy deviation: {round(selected["min_energy_deviation"].value,2)} eV, Max energy deviation: {round(selected["max_energy_deviation"].value,2)} eV')
        self.report(f'Min forces deviation: {round(selected["min_forces_deviation"].value,2)} eV/Å, Max forces deviation: {round(selected["max_forces_deviation"].value,2)} eV/Å')
        self.report(f'Min stress deviation: {round(selected["min_stress_deviation"].value,2)} kbar, Max stress deviation: {round(selected["max_stress_deviation"].value,2)} kbar')
        self.out('committee_evaluation_list', calc.outputs.evaluated_list)       


