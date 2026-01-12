# -*- coding: utf-8 -*-
from aiida.engine import WorkChain, append_, calcfunction, workfunction, if_, while_
from aiida import load_profile
from aiida.orm import Float, Str, StructureData, Int, List, Float, SinglefileData, Bool, Dict, FolderData
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.common import AttributeDict
from ase import Atoms
from aiida.plugins import DataFactory
import numpy as np
from aiida_trains_pot.utils.generate_config import generate_lammps_md_config
import random
from aiida_trains_pot.utils.tools import enlarge_vacuum, error_calibration

load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')
DatasetAugmentationWorkChain    = WorkflowFactory('trains_pot.datasetaugmentation')
TrainingWorkChain               = WorkflowFactory('trains_pot.training')
AbInitioLabellingWorkChain      = WorkflowFactory('trains_pot.labelling')  
ExplorationWorkChain            = WorkflowFactory('trains_pot.exploration')
EvaluationCalculation           = CalculationFactory('trains_pot.evaluation')
PESData                         = DataFactory('pesdata')

PwBaseWorkChain                 = WorkflowFactory('quantumespresso.pw.base')

ALLOWED_ENGINES = ["MACE", "META"]

@calcfunction
def SaveRMSE(rmse):
    """
    A calcfunction to save RMSE values stored in a list of dictionaries as an AiiDA output.

    :param rmse: A list containing dictionaries or JSON-serializable data.
    :return: A List node containing the RMSE values.
    """
    # Convert any AiiDA Dict nodes in the input list to raw dictionaries
    rmse_serializable = [item.get_dict() if isinstance(item, Dict) else item for item in rmse]

    return List(list=rmse_serializable)

@calcfunction
def LammpsFrameExtraction(sampling_time, saving_frequency, thermalization_time=0, check_vacuum=Bool(False), min_vacuum=Float(5.0), target_vacuum=Float(15.0), **trajectories):
    """Extract frames from trajectory."""

    extracted_frames = []      
    for _, trajectory in trajectories.items():

        params = {}
        for inc in trajectory.base.links.get_incoming().all():
            if inc.node.process_type == 'aiida.calculations:lammps.base':
                lammps_id = inc.node.uuid
            if inc.node.process_type == 'aiida.workflows:lammps.base':
                for inc2 in inc.node.base.links.get_incoming().all():
                    if inc2.link_label == 'lammps__parameters':
                        params = Dict(dict=inc2.node).get_dict()
                    elif inc2.link_label == 'lammps__structure':
                        input_structure = enlarge_vacuum(inc2.node.get_ase(), 
                                                        min_vacuum=min_vacuum, 
                                                        target_vacuum=target_vacuum)
                        input_structure_node = inc2.node
                        masses = []
                        symbols = []
                        symbol = input_structure.get_chemical_symbols()
                        for ii, mass in enumerate(input_structure.get_masses()):
                            if mass not in masses:
                                masses.append(mass)
                                symbols.append(symbol[ii])
                            
                        masses, symbols = zip(*sorted(zip(masses, symbols)))
        
        # Convert AiiDA objects to Python types
        thermalization_time_val = thermalization_time.value if hasattr(thermalization_time, 'value') else float(thermalization_time)
        sampling_time_val = sampling_time.value if hasattr(sampling_time, 'value') else float(sampling_time)
        saving_frequency_val = saving_frequency.value if hasattr(saving_frequency, 'value') else int(saving_frequency)
        
        i = int(thermalization_time_val/params['control']['timestep']/saving_frequency_val)
        
        if i == 0: i=1
        while i < trajectory.number_steps:
            step_data = trajectory.get_step_data(i)
            cell = step_data.cell

            extracted_frames.append({'cell': cell.tolist(),
                    'symbols': list(step_data[5]['element']),
                    'positions': [[step_data[5]['x'][jj],step_data[5]['y'][jj],step_data[5]['z'][jj]] for jj, _ in enumerate(step_data[5]['y'])],
                    'input_structure_uuid': str(input_structure_node.uuid),                    
                    'gen_method': str('LAMMPS'),
                    'pbc':trajectory.get_step_structure(i).pbc
                    })
            extracted_frames[-1]['style'] = params['md']['integration']['style']
            extracted_frames[-1]['temp'] = params['md']['integration']['constraints']['temp']
            extracted_frames[-1]['timestep'] = params['control']['timestep']
            extracted_frames[-1]['id_lammps'] = lammps_id

            i = i + int(sampling_time_val/params['control']['timestep']/saving_frequency_val)

    pes_extracted_frames = PESData(extracted_frames)    
    return {'explored_dataset': pes_extracted_frames}

@calcfunction
def SelectToLabel(evaluated_dataset, thr_energy, thr_forces, thr_stress, max_frames=None):
    """Select configurations to label."""
    if max_frames:
        max_frames = max_frames.value
    selected_dataset = []
    energy_deviation = []
    forces_deviation = []
    stress_deviation = []
    loss = []
    for config in evaluated_dataset:
        energy_deviation.append(config['energy_deviation'])
        forces_deviation.append(config['forces_deviation'])
        stress_deviation.append(config['stress_deviation'])
        if config['energy_deviation'] > thr_energy or config['forces_deviation'] > thr_forces or config['stress_deviation'] > thr_stress:
            selected_dataset.append(config)
            if max_frames:
                loss.append(config['energy_deviation']/thr_energy + config['forces_deviation']/thr_forces + config['stress_deviation']/thr_stress)
    if max_frames:
        if len(selected_dataset) > max_frames:
            random.shuffle(selected_dataset)
            selected_dataset = selected_dataset[:max_frames]
    pes_selected_dataset = PESData(selected_dataset)
    return {'selected_dataset':pes_selected_dataset, 'min_energy_deviation':Float(min(energy_deviation)), 'max_energy_deviation':Float(max(energy_deviation)), 'min_forces_deviation':Float(min(forces_deviation)), 'max_forces_deviation':Float(max(forces_deviation)), 'min_stress_deviation':Float(min(stress_deviation)), 'max_stress_deviation':Float(max(stress_deviation))}


def validate_engine(value, _):
    if value.value not in ALLOWED_ENGINES:
        return f"Invalid training engine: {value.value}. Must be one of {ALLOWED_ENGINES}."
    return None

class TrainsPotWorkChain(WorkChain):
    """WorkChain to launch LAMMPS calculations."""


   
    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""

        ######################################################
        ##                 DEFAULT VALUES                   ##
        ######################################################
        DEFAULT_thr_energy                      = Float(0.001)
        DEFAULT_thr_forces                      = Float(0.1)
        DEFAULT_thr_stress                      = Float(0.001)

        DEFAULT_max_selected_frames             = Int(1000)
        DEFAULT_random_input_structures_lammps  = Bool(True)
        DEFAULT_num_random_structures_lammps    = Int(20)

        DEFAULT_thermalization_time             = Float(0.0)
        DEFAULT_sampling_time                   = Float(1.0)

        DEFAULT_max_loops                       = Int(10)

        DEFAULT_do_dataset_augmentation         = Bool(True)
        DEFAULT_do_ab_initio_labelling          = Bool(True)
        DEFAULT_training_engine                 = Str("MACE")
        DEFAULT_do_training                     = Bool(True)
        DEFAULT_do_exploration                  = Bool(True)

        DEFAULT_check_vacuum                    = Bool(True)
        ######################################################
        
        
        
        super().define(spec)
        spec.input('do_dataset_augmentation', valid_type=Bool, default=lambda: DEFAULT_do_dataset_augmentation, help='Do data generation', required=False)
        spec.input('do_ab_initio_labelling', valid_type=Bool, default=lambda: DEFAULT_do_ab_initio_labelling, help='Do ab_initio_labelling calculations', required=False)
        spec.input('training_engine', valid_type=Str, default=lambda: DEFAULT_training_engine, help=f"Training engine (allowed values: {ALLOWED_ENGINES})", required=False, validator=validate_engine)
        #spec.input('training_engine', valid_type=Str, default=lambda: DEFAULT_training_engine, help='Training engine', required=False)
        spec.input('do_training', valid_type=Bool, default=lambda: DEFAULT_do_training, help='Do MACE calculations', required=False)
        spec.input('do_exploration', valid_type=Bool, default=lambda: DEFAULT_do_exploration, help='Do exploration calculations', required=False)
        spec.input('max_loops', valid_type=Int, default=lambda: DEFAULT_max_loops, help='Maximum number of active learning workflow loops', required=False)

        spec.input('random_input_structures_lammps', valid_type=Bool, help='If true, input structures for LAMMPS are randomly selected from the dataset', default=lambda: DEFAULT_random_input_structures_lammps, required=False)
        spec.input('num_random_structures_lammps', valid_type=Int, help='Number of random structures for LAMMPS', default=lambda: DEFAULT_num_random_structures_lammps, required=False)
        spec.input('lammps_input_structures', valid_type=PESData, help='Input structures for lammps, if not specified input structures are used', required=False)
        spec.input('dataset', valid_type=PESData, help='Dataset containing labelled structures and structures to be labelled', required=True)

        spec.input_namespace('models_lammps', valid_type=SinglefileData, help='MACE potential for md exploration', required=False)
        spec.input_namespace('models_ase', valid_type=SinglefileData, help='MACE potential for Evaluation', required=False) 
        spec.input('exploration.parameters', valid_type=Dict, help='List of parameters for md exploration', required=False)
        spec.input('explored_dataset', valid_type=PESData, help='List of structures from exploration', required=False)
        
        # spec.input('potential', valid_type=SinglefileData, help='MACE potential for exploration', required=False)

        spec.input('frame_extraction.sampling_time', valid_type=Float, help='Correlation time for frame extraction', required=False, default=lambda: DEFAULT_sampling_time)
        spec.input('frame_extraction.thermalization_time', valid_type=Float, default=lambda : DEFAULT_thermalization_time, help='Thermalization time for exploration', required=False)

        spec.input('thr_energy', valid_type=Float, help='Threshold for energy', required=True, default=lambda: DEFAULT_thr_energy)
        spec.input('thr_forces', valid_type=Float, help='Threshold for forces', required=True, default=lambda: DEFAULT_thr_forces)
        spec.input('thr_stress', valid_type=Float, help='Threshold for stress', required=True, default=lambda: DEFAULT_thr_stress)
        spec.input('max_selected_frames', valid_type=Int, help='Maximum number of frames to be selected for labelling per iteration', required=False, default=lambda: DEFAULT_max_selected_frames) 

        spec.input('check_vacuum', valid_type=Bool, default=lambda:DEFAULT_check_vacuum, help='Check vacuum in the explored structures', required=False)
        spec.input('vacuum.min_vacuum', valid_type=Float, help='Minimum vacuum size to consider for enlarging, if not specified NNIP cutoff will be used', required=False)
        spec.input('vacuum.target_vacuum', valid_type=Float, help='Target vacuum size after enlarging, if not specified dataset_augmentation vacuum value will be used', required=False)

        spec.inputs.validator = cls.validate_inputs

        spec.expose_inputs(DatasetAugmentationWorkChain, namespace="dataset_augmentation", exclude=('structures'))
        spec.expose_inputs(AbInitioLabellingWorkChain, namespace="ab_initio_labelling",  exclude=('unlabelled_dataset'), namespace_options={'validator': None})
        spec.expose_inputs(TrainingWorkChain, namespace="training", exclude=('dataset'), namespace_options={'validator': None})
        spec.expose_inputs(ExplorationWorkChain, namespace="exploration", exclude=('potential_lammps', 'lammps_input_structures','sampling_time'), namespace_options={'validator': None})
        spec.expose_inputs(EvaluationCalculation, namespace="committee_evaluation", exclude=('mace_potentials', 'datasetlist'))
        
        spec.output("dataset", valid_type=PESData, help="Final dataset containing all structures labelled and selected to be labelled")
        spec.output_namespace("models_ase", required = False, valid_type=SinglefileData, help="Last committee of trained potentials compiled for ASE")
        spec.output_namespace("models_lammps", required = False, valid_type=SinglefileData, help="Last committee of trained potentials compiled for LAMMPS")
        spec.output_namespace("checkpoints", required = False, valid_type=FolderData, help="Last checkpoints of trained potentials")
        spec.output("RMSE", required = False, valid_type=List, help="RMSE on the final dataset computed with the last committee of potentials")

        spec.exit_code(200, "NO_LABELLED_STRUCTURES", message="No labelled structures in the dataset.",)
        spec.exit_code(201, "MISSING_PSEUDOS", message="Missing pseudopotentials for some atomic species in the input dataset.",)
        
        spec.exit_code(308, "LESS_THAN_2_POTENTIALS", message="Calculation didn't produce more tha 1 expected potentials.",)
        spec.exit_code(309, "NO_MD_CALCULATIONS", message="Calculation didn't produce any MD calculations.",)
        spec.exit_code(310, "EMPTY_EXPLORATION_DATASET", message="The exploration dataset is empty.",)

        
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
                if_(cls.do_exploration)(
                    cls.exploration,
                    cls.finalize_exploration,
                    cls.exploration_frame_extraction),
                if_(cls.do_evaluation)(
                cls.run_committee_evaluation,
                cls.finalize_committee_evaluation),
            ),
            cls.finalize            
        )

    @classmethod
    def validate_inputs(cls, inputs, namespace):
        """Validate the top-level inputs."""
        if 'check_vacuum' not in inputs:
            inputs['check_vacuum'] = Bool(True)
        if inputs['check_vacuum']:
            if "vacuum" not in inputs or "min_vacuum" not in inputs['vacuum']:
                inputs["vacuum"] = {}
                try:
                    inputs["vacuum"]["min_vacuum"] = Float(inputs['training']['mace']['train']['mace_config'].get_dict()['r_max'])
                    print(f"Warning: `vacuum.min_vacuum` not specified, using user-defined MACE cutoff radius value of {inputs['vacuum']['min_vacuum']} Ang.")
                except:
                    inputs["vacuum"]["min_vacuum"] = Float(5.0) # Default value of MACE 0.3.13
                    print(f"Warning: `vacuum.min_vacuum` not specified, using default MACE default cutoff radius value of {inputs['vacuum']['min_vacuum']} Ang.")
            if "vacuum" not in inputs or "target_vacuum" not in inputs['vacuum']:
                try:
                    inputs["vacuum"]["target_vacuum"] = inputs['dataset_augmentation']['vacuum']
                except:
                    return "The `vacuum.target_vacuum` or `dataset_augmentation.vacuum` input is required when using the 'check_vacuum' option."
                print(f"Warning: `vacuum.target_vacuum` not specified, using `dataset_augmentation.vacuum` value of {inputs['vacuum']['target_vacuum']} Ang.")
        cls.inputs = AttributeDict(inputs)




    @classmethod
    def get_builder(cls, dataset, abinitiolabeling_code, md_code, training_code=None, abinitiolabeling_protocol=None, pseudo_family=None, md_protocol=None, **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param dataset: The dataset to use for the calculation.
        :param abinitiolabeling_protocol: The protocol to use for the ab initio labelling calculation.
        :param abinitiolabeling_code: The code to use for the ab initio labelling calculation.
        :param pseudo_family: The pseudo family to use for the calculation.
        :param md_protocol: The protocol to use for the MD calculation.
        :param kwargs: Additional keyword arguments to pass to the builder.

        :return: A builder prepopulated with inputs selected according to the chosen protocol.
        """
        builder = super().get_builder(**kwargs)
        builder.dataset = dataset

        ### Quantum ESPRESSO ###
        qe_protocol = abinitiolabeling_protocol or 'stringent'

        atomic_species = dataset.get_atomic_species()
        fictitious_structure = StructureData(ase=Atoms(atomic_species))
        if pseudo_family is not None:
            overrides = {'pseudo_family': pseudo_family}
        else:
            overrides = {}
        qe_builder = PwBaseWorkChain.get_builder_from_protocol(protocol=qe_protocol, code=abinitiolabeling_code, structure=fictitious_structure, overrides=overrides)
        builder.ab_initio_labelling.quantumespresso = qe_builder

        ### LAMMPS ###
        if md_protocol not in ['vdw_d2', None]:
            raise ValueError(f"MD protocol {md_protocol} not found.")
        if md_protocol == 'vdw_d2':
            builder.exploration.potential_pair_style = Str('hybrid/overlay')
        builder.exploration.md.lammps.code  = md_code
        builder.exploration.params_list     = generate_lammps_md_config()
        builder.exploration.protocol        = md_protocol
        builder.exploration.parameters      = Dict({"control":{"timestep": 0.001}})

        return builder

    def do_dataset_augmentation(self): return bool(self.ctx.do_dataset_augmentation)
    def do_ab_initio_labelling(self): return bool(self.ctx.do_ab_initio_labelling)
    def do_training(self): return bool(self.ctx.do_training)
    def do_exploration(self): return bool(self.ctx.do_exploration)
    def do_evaluation(self):
        return bool('explored_dataset' in self.ctx)
    def check_iteration(self):
        if self.ctx.iteration > 0:
            self.ctx.do_dataset_augmentation = False
            self.ctx.do_ab_initio_labelling = True
            self.ctx.do_training = True
            self.ctx.do_exploration = True
        self.ctx.iteration += 1
        return self.ctx.iteration < self.inputs.max_loops+1

    def initialization(self):
        """Initialize variables."""
        self.ctx.thr_energy = self.inputs.thr_energy
        self.ctx.thr_forces = self.inputs.thr_forces
        self.ctx.thr_stress = self.inputs.thr_stress
        if 'max_selected_frames' in self.inputs:
            self.ctx.max_frames = self.inputs.max_selected_frames
        else:
            self.ctx.max_frames = None

        self.ctx.rmse = []       
        self.ctx.iteration = 0
        if 'dataset' in self.inputs:
            self.ctx.dataset = self.inputs.dataset
        else:
            self.ctx.dataset = PESData()
        self.ctx.do_dataset_augmentation = self.inputs.do_dataset_augmentation
        self.ctx.do_ab_initio_labelling = self.inputs.do_ab_initio_labelling
        self.ctx.do_training = self.inputs.do_training
        self.ctx.do_exploration = self.inputs.do_exploration
        if not self.ctx.do_ab_initio_labelling and self.ctx.do_training:
            if "dataset" in self.inputs:
                if self.inputs.dataset.len_labelled > 0:
                    self.ctx.dataset = self.inputs.dataset
                else:
                    return self.exit_codes.NO_LABELLED_STRUCTURES
                

        if not self.ctx.do_training:
            self.ctx.potentials_lammps = []
            self.ctx.potentials_ase = []
            self.ctx.potential_checkpoints = []
            if "models_lammps" in self.inputs:
                for _, pot in self.inputs.models_lammps.items():
                    self.ctx.potentials_lammps.append(pot)
            if "models_ase" in self.inputs:
                for _, pot in self.inputs.models_ase.items():
                    self.ctx.potentials_ase.append(pot)
            if "checkpoints" in self.inputs:
                for _, pot in self.inputs.training.checkpoints.items():
                    self.ctx.potential_checkpoints.append(pot)
        if not self.ctx.do_exploration and 'explored_dataset' in self.inputs:
            if len(self.inputs.explored_dataset) > 0:
                self.ctx.explored_dataset = self.inputs.explored_dataset

        if 'lammps_input_structures' in self.inputs:
            self.ctx.lammps_input_structures = self.inputs.lammps_input_structures
        else:
            self.ctx.lammps_input_structures = PESData([atm for atm in self.inputs.dataset.get_ase_list()])

        atomic_species = self.ctx.dataset.get_atomic_species()
        for specie in atomic_species:
            if specie not in self.inputs.ab_initio_labelling.quantumespresso.pw.pseudos.keys():
                return self.exit_codes.MISSING_PSEUDOS
    def dataset_augmentation(self):
        """Generate data for the dataset."""
        
        inputs = self.exposed_inputs(DatasetAugmentationWorkChain, namespace="dataset_augmentation")
        inputs['structures'] = self.ctx.dataset

        future = self.submit(DatasetAugmentationWorkChain, **inputs)
        self.report(f'launched lammps calculation <{future.pk}>')
        self.to_context(dataset_augmentation = future)
    
    def ab_initio_labelling(self):
        """Run ab_initio_labelling calculations."""

        # Set up the inputs for LoopingLabellingWorkChain
        inputs = self.exposed_inputs(AbInitioLabellingWorkChain, namespace="ab_initio_labelling")
        inputs.unlabelled_dataset = self.ctx.dataset.get_unlabelled()
                

        # Submit LoopingLabellingWorkChain
        future = self.submit(AbInitioLabellingWorkChain, **inputs)

        self.report(f'Launched AbInitioLabellingWorkChain with ase_list <{future.pk}>')
        self.to_context(ab_initio_labelling = future)

    def training(self):
        """Run training calculations."""

        inputs = self.exposed_inputs(TrainingWorkChain, namespace="training")
        inputs.dataset = self.ctx.dataset.get_labelled()
        inputs.engine = self.inputs.training_engine
        if 'potential_checkpoints' in self.ctx:
            inputs['checkpoints'] = {f"chkpt_{ii+1}": self.ctx.potential_checkpoints[-ii] for ii in range(min(len(self.ctx.potential_checkpoints), self.inputs.training.num_potentials.value))}
      
      
        future = self.submit(TrainingWorkChain, **inputs)

        self.report(f'Launched TrainingWorkChain with dataset_list <{future.pk}>')
        self.to_context(training = future)            

    def exploration(self):
        """Run exploration."""
        inputs = self.exposed_inputs(ExplorationWorkChain, namespace="exploration")
        inputs.potential_lammps = self.ctx.potentials_lammps[-1]
        
        if "random_input_structures_lammps" in self.inputs:
            if self.inputs.random_input_structures_lammps:
                if 'input_lammps_dataset' in self.ctx:
                    self.ctx.lammps_input_structures = self.ctx.input_lammps_dataset
                else:
                    self.ctx.lammps_input_structures = self.ctx.dataset

        
        # Select random input structures for LAMMPS avoiding isolated atoms
        discarded = set()
        selected = []
        num_structures = self.inputs.num_random_structures_lammps.value
        while len(selected) < num_structures:
            # If choosed all non-discarded unique values, break
            remaining_capacity = len(self.ctx.lammps_input_structures) - len(discarded)
            if remaining_capacity == len(selected):
                self.report(f'Only {len(selected)} random input structures for LAMMPS are selected ({num_structures} where requested).')
                break

            x = random.choice(range(len(self.ctx.lammps_input_structures)))
            if len(self.ctx.lammps_input_structures.get_ase_item(x)) < 2:
                discarded.add(x)
                continue  # reject isolated atoms
            if x in selected:
                continue  # reject duplicate
            selected.append(x)
        
        self.ctx.lammps_input_structures = PESData([self.ctx.lammps_input_structures.get_item(key) for key in selected])
            
        inputs.lammps_input_structures = self.ctx.lammps_input_structures
        inputs.sampling_time = self.inputs.frame_extraction.sampling_time

        future = self.submit(ExplorationWorkChain, **inputs)

        self.report(f'Launched ExplorationWorkChain with dataset_list <{future.pk}>')
        self.to_context(exploration = future)
    
    def exploration_frame_extraction(self):
        """Run exploration frame extraction."""
        # for _, trajectory in self.ctx.trajectories.items():        
        parameters=AttributeDict(self.inputs.exploration.parameters)
        dump_rate = int(self.inputs.frame_extraction.sampling_time/parameters.control.timestep)
        explored_dataset = LammpsFrameExtraction(self.inputs.frame_extraction.sampling_time,
                                dump_rate,
                                thermalization_time = self.inputs.frame_extraction.thermalization_time,
                                check_vacuum = self.inputs.check_vacuum,
                                min_vacuum = self.inputs.vacuum.min_vacuum,
                                target_vacuum = self.inputs.vacuum.target_vacuum,
                                **self.ctx.trajectories)['explored_dataset']
        if len(explored_dataset) == 0:
            self.finalize()
            return self.exit_codes.EMPTY_EXPLORATION_DATASET
        self.ctx.explored_dataset = explored_dataset
      
    def run_committee_evaluation(self):
        inputs = self.exposed_inputs(EvaluationCalculation, namespace="committee_evaluation")
        if len(self.ctx.potentials_ase) == 0:
            self.ctx.potentials_ase = self.ctx.potentials_lammps
        inputs['ase_potentials'] = {f"pot_{ii}": self.ctx.potentials_ase[ii] for ii in range(len(self.ctx.potentials_ase))}
        inputs['datasets'] = {"labelled": self.ctx.dataset, "exploration": self.ctx.explored_dataset}

        future = self.submit(EvaluationCalculation, **inputs)
        self.to_context(committee_evaluation = future)  

    def finalize_dataset_augmentation(self):
        """Finalize dataset augmentation."""
        self.ctx.dataset += self.ctx.dataset_augmentation.outputs.structures.global_structures
        self.ctx.input_lammps_dataset = self.ctx.dataset
    
    def finalize_ab_initio_labelling(self):
        self.ctx.dataset = self.ctx.dataset.get_labelled() + self.ctx.ab_initio_labelling.outputs.ab_initio_labelling_data
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
                       
        self.ctx.dataset = self.ctx.training.outputs.global_splitted       


    def finalize_exploration(self):

        if len(self.ctx.exploration.outputs.md) < 1:
            return self.exit_codes.NO_MD_CALCULATIONS
        
        self.ctx.trajectories = {}        
        for ii, calc in enumerate(self.ctx.exploration.outputs.md.values()):            
            
            for key, value in calc.items():
                if key == 'trajectories':
                    self.ctx.trajectories[f'exploration_{ii}'] = value
        self.ctx.exploration = []          
    
    def finalize_committee_evaluation(self):
        calc = self.ctx.committee_evaluation
        self.ctx.thr_energy, self.ctx.thr_forces, self.ctx.thr_stress = error_calibration(calc.outputs.evaluated_datasets.labelled, self.inputs.thr_energy, self.inputs.thr_forces, self.inputs.thr_stress)
        selected = SelectToLabel(calc.outputs.evaluated_datasets.exploration, self.ctx.thr_energy, self.ctx.thr_forces, self.ctx.thr_stress, self.ctx.max_frames)
        self.ctx.dataset += selected['selected_dataset']
        #self.ctx.rmse.append(calc.outputs.rmse)
        self.ctx.rmse.append(calc.outputs.rmse.labelled.get_dict())

        self.report(f'Structures selected for labelling: {len(selected["selected_dataset"])}/{len(calc.outputs.evaluated_datasets.exploration)}')
        self.report(f'Min energy deviation: {round(selected["min_energy_deviation"].value,2)} eV, Max energy deviation: {round(selected["max_energy_deviation"].value,2)} eV')
        self.report(f'Min forces deviation: {round(selected["min_forces_deviation"].value,2)} eV/Å, Max forces deviation: {round(selected["max_forces_deviation"].value,2)} eV/Å')
        self.report(f'Min stress deviation: {round(selected["min_stress_deviation"].value,2)} kbar, Max stress deviation: {round(selected["max_stress_deviation"].value,2)} kbar')

    def finalize(self):
        if 'rmse' in self.ctx:
            self.out('RMSE', SaveRMSE(self.ctx.rmse))
        self.out('dataset', self.ctx.dataset)
        if 'potentials_ase' in self.ctx:
            self.out('models_ase', {f"model_{ii+1}": pot for ii, pot in enumerate(self.ctx.potentials_ase)})
        if 'potentials_lammps' in self.ctx:
            self.out('models_lammps', {f"model_{ii+1}": pot for ii, pot in enumerate(self.ctx.potentials_lammps)})
        if 'potential_checkpoints' in self.ctx:
            self.out('checkpoints', {f"model_{ii+1}": pot for ii, pot in enumerate(self.ctx.potential_checkpoints)})