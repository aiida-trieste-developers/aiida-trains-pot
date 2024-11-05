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
from aiida.plugins import DataFactory
import tempfile
from ase.io.lammpsrun import read_lammps_dump_text
from io import StringIO
import numpy as np
from pathlib import Path
import random
import itertools
import time
import os
import io
load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')
DatasetAugmentationWorkChain    = WorkflowFactory('trains_pot.datasetaugmentation')
PwBaseWorkChain                 = WorkflowFactory('quantumespresso.pw.base')
MaceWorkChain                   = WorkflowFactory('trains_pot.macetrain')
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
def WriteLabelledList(non_labelled_structures, **labelled_data):
    labelled_list = []
    for key, value in labelled_data.items():
        labelled_list.append(non_labelled_structures.get_list()[int(key.split('_')[1])])
        labelled_list[-1]['dft_energy'] = float(value['output_parameters'].dict.energy)
        labelled_list[-1]['dft_forces'] = value['output_trajectory'].get_array('forces')[0].tolist()
        labelled_list[-1]['dft_stress'] = value['output_trajectory'].get_array('stress')[0].tolist()

    pes_labelled_list = PESData()    
    pes_labelled_list.set_list(labelled_list)    
    return pes_labelled_list

@calcfunction
def SplitDataset(dataset):
    """Divide dataset into training, validation and test sets."""
    # data = self.inputs.dataset_list.get_list()
    data = dataset.get_list()

    exclude_list = ["energy", "cell", "stress", "forces", "symbols", "positions"]
    # Define a function to extract the grouping key
    def check_esclude_list(string):
        for el in exclude_list:
             if el in string:
                 return False
        return True
    
    def get_grouping_key(d):
        return tuple((k, v) for k, v in d.items() if check_esclude_list(k))

    # Sort the data based on the grouping key
    sorted_data = sorted(data, key=get_grouping_key)

    # Group the sorted data by the grouping key
    grouped_data = itertools.groupby(sorted_data, key=get_grouping_key)

    # Iterate over the groups and print the group key and the list of dictionaries in each group
    training_set = []
    validation_set = []
    test_set = []

    for _, group in grouped_data:
    # Calculate the number of elements for each set
        group_list = list(group)

        if group_list[0]['gen_method'] == "INPUT_STRUCTURE" or group_list[0]['gen_method'] == "ISOLATED_ATOM" or len(group_list[0]['positions']) == 1 or group_list[0]['gen_method'] == "EQUILIBRIUM":
                training_set += group_list
                continue
        elif 'set' in group_list[0].keys():
            if group_list[0]['set'] == 'TRAINING':
                training_set += group_list
                continue
            elif group_list[0]['set'] == 'VALIDATION':
                validation_set += group_list
                continue
            elif group_list[0]['set'] == 'TEST':
                test_set += group_list
                continue
        total_elements = len(group_list)
        training_size = int(0.8 * total_elements)
        test_size = int(0.1 * total_elements)
        validation_size = total_elements - training_size - test_size
        
        random.seed(int(time.time()))
        _ = random.shuffle(group_list)


        # Split the data into sets
        training_set += group_list[:training_size]
        validation_set += group_list[training_size:training_size+validation_size]
        test_set += group_list[training_size+validation_size:]

    for ii in range(len(training_set)):
        training_set[ii]['set'] = 'TRAINING'
    for ii in range(len(validation_set)):
        validation_set[ii]['set'] = 'VALIDATION'
    for ii in range(len(test_set)):
        test_set[ii]['set'] = 'TEST'

    pes_training_set = PESData()    
    pes_training_set.set_list(training_set)    

    pes_validation_set = PESData()    
    pes_validation_set.set_list(validation_set)  

    pes_test_set = PESData()    
    pes_test_set.set_list(test_set)  

    pes_global_splitted = PESData()    
    pes_global_splitted.set_list(training_set+validation_set+test_set)  
    
    return {"train_set":pes_training_set, "validation_set":pes_validation_set, "test_set":pes_test_set, "global_splitted":pes_global_splitted}

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
            extracted_frames[-1]['style'] = params['md_exploration']['integration']['style']
            extracted_frames[-1]['temp'] = params['md_exploration']['integration']['constraints']['temp']
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
        spec.input("training.num_potentials", valid_type=Int, default=lambda:Int(1), required=False)
        
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
        spec.expose_inputs(PwBaseWorkChain, namespace="ab_initio_labelling", exclude=('pw.structure',), namespace_options={'validator': None})
        spec.expose_inputs(MaceWorkChain, namespace="training",  exclude=('mace.training_set', 'mace.validation_set', 'mace.test_set'), namespace_options={'validator': None})
        spec.expose_inputs(LammpsWorkChain, namespace="md_exploration", exclude=('lammps.structure', 'lammps.potential','lammps.parameters'), namespace_options={'validator': None})
        spec.expose_inputs(EvaluationCalculation, namespace="committee_evaluation", exclude=('mace_potentials', 'datasetlist'))
        # spec.expose_inputs(FrameExtractionWorkChain, namespace="frame_extraction", exclude=('trajectories', 'input_structure', 'dt', 'saving_frequency'))

        spec.input_namespace("structures", valid_type=StructureData, required=True)

        spec.output("ab_initio_labelling.labelled_list", valid_type=PESData, help="List of configurations labelled via ab_initio_labelling")
        spec.output("md_exploration.lammps_extracted_list", valid_type=PESData, help="List of extracted frames from md_exploration trajectories")
        spec.output("committee_evaluation_list", valid_type=PESData, help="List of committee evaluated configurations")
        spec.output_namespace("md_exploration", dynamic=True, help="md_exploration outputs")
        spec.output_namespace("training", dynamic=True, help="Training outputs")
        spec.expose_outputs(DatasetAugmentationWorkChain, namespace="data_set_augmentation")        
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
        self.ctx.config = 0
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

        self.ctx.config += 1
        
        if self.ctx.iteration > 1:
            ase_list = self.ctx.committee_evaluation_list.get_ase_list()
        else:
            if self.ctx.do_data_set_augmentation:
                ase_list = self.ctx.data_set_augmentation.outputs.structure_lists.global_structure_list.get_ase_list()            
            else:
                ase_list = self.inputs.non_labelled_list.get_ase_list()


        for _, structure in enumerate(ase_list):
            self.ctx.config += 1
            str_data = StructureData(ase=structure)
            default_inputs = {'CONTROL': {'calculation': 'scf', 'tstress': True, 'tprnfor': True}}

            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='ab_initio_labelling'))
            inputs.pw.structure = str_data
            inputs.metadata.call_link_label = f'ab_initio_labelling_config_{self.ctx.config}'
            
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
            self.to_context(ab_initio_labelling_calculations=append_(future))

    def training(self):
        """Run training calculations."""
        dataset_list = PESData()
        if self.ctx.do_ab_initio_labelling:
            dataset_list = self.ctx.labelled_list
        else:
            dataset_list = self.inputs.labelled_list

        split_datasets = SplitDataset(dataset_list)
        train_set = split_datasets["train_set"]
        validation_set = split_datasets["validation_set"]
        test_set = split_datasets["test_set"]

        self.ctx.global_splitted=split_datasets["global_splitted"]
        
        self.report(f"Training set size: {len(train_set.get_list())}")
        self.report(f"Validation set size: {len(validation_set.get_list())}")
        self.report(f"Test set size: {len(test_set.get_list())}")     
      

        inputs = self.exposed_inputs(MaceWorkChain, namespace="training")

        inputs.mace["training_set"] = train_set
        inputs.mace["validation_set"] = validation_set
        inputs.mace["test_set"] = test_set

        
        if self.ctx.iteration > 1:
            inputs['checkpoints'] = {f"chkpt_{ii+1}": self.ctx.checkpoints[-ii] for ii in range(min(len(self.ctx.checkpoints), self.inputs.training.num_potentials.value))}
            inputs.mace['restart'] = Bool(True)    

        if 'checkpoints' in inputs:
            chkpts = list(dict(inputs.checkpoints).values())

        for ii in range(self.inputs.training.num_potentials.value):            
            if 'checkpoints' in self.inputs and ii < len(chkpts):
                inputs.mace["checkpoints"] = chkpts[ii]
            
            inputs.mace["index_pot"] = Int(ii)
            future = self.submit(MaceWorkChain, **inputs)
            self.to_context(mace_wc = append_(future))        
        pass

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
        ab_initio_labelling_data = {}
        for ii, calc in enumerate(self.ctx.ab_initio_labelling_calculations):
            if calc.exit_status == 0:
                ab_initio_labelling_data[f'abinitiolabelling_{ii}'] = {
                    'output_parameters': calc.outputs.output_parameters,
                    'output_trajectory': calc.outputs.output_trajectory
                    }
        if self.ctx.do_data_set_augmentation:
            labelled_list = WriteLabelledList(non_labelled_structures = self.ctx.data_set_augmentation.outputs.structure_lists.global_structure_list, **ab_initio_labelling_data)
        elif self.ctx.iteration > 1:
            labelled_list = WriteLabelledList(non_labelled_structures = self.ctx.committee_evaluation_list.get_list(), **ab_initio_labelling_data)
        else:
            labelled_list = WriteLabelledList(non_labelled_structures = self.inputs.non_labelled_list, **ab_initio_labelling_data)
        
        self.ctx.labelled_list += labelled_list
        self.out('ab_initio_labelling.labelled_list', labelled_list)
        self.ctx.ab_initio_labelling_calculations = []

    def finalize_training(self):
        self.ctx.potentials = []
        self.ctx.potentials_lammps = []
        self.ctx.checkpoints = []
        potentials = {}
        for ii, calc in enumerate(self.ctx.mace_wc):
            potentials[f'mace_{ii}']={}
            for el in calc.outputs:
                
                if el == 'swa_ase_model':
                    self.ctx.potentials.append(calc.outputs[el])
                elif el == 'checkpoints':
                    self.ctx.checkpoints.append(calc.outputs[el])
                elif el == 'swa_model_lammps':
                    self.ctx.potentials_lammps.append(calc.outputs[el])
            
            
                potentials[f'mace_{ii}'][el] = calc.outputs[el]
                
            self.out('training', potentials)
           
        self.ctx.labelled_list = self.ctx.global_splitted
        # self.out_many(self.exposed_outputs(self.ctx.mace_wc, MaceWorkChain, namespace="mace"))

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


