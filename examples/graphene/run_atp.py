from aiida.orm import load_code, load_node, load_computer, Str, Dict, List, Int, Bool, Float
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory, DataFactory
from ase.io import read
import yaml
import os
from aiida_trains_pot.utils.restart import models_from_trainingwc,  models_from_aiidatrainspotwc
from aiida_trains_pot.utils.generate_config import generate_lammps_md_config
load_profile()

PESData     = DataFactory('pesdata')
KpointsData = DataFactory("core.array.kpoints")
TrainsPot   = WorkflowFactory('trains_pot.workflow')

####################################################################
#                     START MACHINE PARAMETERS                     #
####################################################################


#QE_code                 = load_code('qe7.2-pw@leo5_scratch_bind')
#MACE_train_code         = load_code('mace0312@leo5_scratch')
#MACE_preprocess_code    = load_code('mace_preprocess@leo5_scratch')
#MACE_postprocess_code   = load_code('mace_postprocess@leo5_scratch')
#LAMMPS_code             = load_code('lmp4mace@leo5_scratch')
#EVALUATION_code         = load_code('cep0312')

EVALUATION_computer     = load_computer('leo1_scratch')

QE_code                 = load_code('qe7.2-pw@leo1_scratch_bind')

META_train_code         = load_code('metatrain@leo1_scratch_mace')
MACE_train_code         = load_code('mace_train_func_312@leo1_scratch_mace')
MACE_preprocess_code    = load_code('mace_preprocess@leo1_scratch_mace')
MACE_postprocess_code   = load_code('mace_postprocess@leo1_scratch_mace')
#LAMMPS_code             = load_code('lmp4mace@leo1_scratch')
LAMMPS_code             = load_code('lmp4meta@leo1_scratch')
EVALUATION_code         = load_code('committee_evaluation_portable_312')


QE_machine = {
'time'                             : "00:05:00",
'nodes'                            : 1,
'gpu'                              : "1",
'taskpn'                           : 1,
'cpupt'                            : "8",
'mem'                              : "70GB",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'qos'                              : "boost_qos_dbg"
}

MACE_machine = {
'time'                             : "00:30:00",
'nodes'                            : 1,
'gpu'                              : "1",
'taskpn'                           : 1,
'cpupt'                            : "8",
'mem'                              : "30GB",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'qos'                              : "boost_qos_dbg"
}

META_machine = {
'time'                             : "00:05:00",
'nodes'                            : 1,
'gpu'                              : "1",
'taskpn'                           : 1,
'cpupt'                            : "8",
'mem'                              : "30GB",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'qos'                              : "boost_qos_dbg"
}

LAMMPS_machine = {
'time'                             : "00:30:00",
'nodes'                            : 1,
'gpu'                              : "1",
'taskpn'                           : 1,
'cpupt'                            : "8",
'mem'                              : "30GB",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'qos'                              : "boost_qos_dbg"
}

EVALUATION_machine = {
 'time'                             : "00:30:00",
 'nodes'                            : 1,
 'gpu'                              : "1",
 'taskpn'                           : 1,
 'cpupt'                            : "8",
 'mem'                              : "30GB",
 'account'                          : "CNHPC_1491920",
 'partition'                        : "boost_usr_prod",
 'qos'                              : "boost_qos_dbg"
 }

####################################################################
#                      END MACHINE PARAMETERS                      #
####################################################################

def get_memory(mem):
    if mem.find('MB') != -1:
        mem = int(mem.replace('MB',''))*1024
    elif mem.find('GB') != -1:
        mem = int(mem.replace('GB',''))*1024*1024
    elif mem.find('KB') != -1:
        mem = int(mem.replace('KB',''))
    return mem

def get_time(time):
    time = time.split(':')
    time_sec=int(time[0])*3600+int(time[1])*60+int(time[2])
    return time_sec

QE_mem = get_memory(QE_machine['mem'])
QE_time = get_time(QE_machine['time'])

MACE_mem = get_memory(MACE_machine['mem'])
MACE_time = get_time(MACE_machine['time'])

META_mem = get_memory(META_machine['mem'])
META_time = get_time(META_machine['time'])

LAMMPS_mem = get_memory(LAMMPS_machine['mem'])
LAMMPS_time = get_time(LAMMPS_machine['time'])

EVALUATION_mem = get_memory(EVALUATION_machine['mem'])
EVALUATION_time = get_time(EVALUATION_machine['time'])

script_dir = os.path.dirname(os.path.abspath(__file__))

###############################################
# Input structures
###############################################

input_structures = PESData([read(os.path.join(script_dir, 'gr8x8.xyz'))])

###############################################
# Setup TrainsPot worflow
###############################################

builder                             = TrainsPot.get_builder(abinitiolabeling_code     = QE_code,
                                                            abinitiolabeling_protocol = 'fast',
                                                            pseudo_family             = 'SSSP/1.3/PBE/efficiency',
                                                            md_code                   = LAMMPS_code,
                                                            #md_protocol               = 'vdw_d2',
                                                            #dataset                   = input_structures,
                                                            dataset                   = load_node(1810107),
                                                            )
builder.do_dataset_augmentation     = Bool(False)
builder.do_ab_initio_labelling      = Bool(False)
builder.training_engine             = Str("META")
builder.do_training                 = Bool(True)
builder.do_exploration              = Bool(True)
builder.max_loops                   = Int(2)

## Additional inputs for restart from previous runs or to start with a previous dataset and/or previous MACE potentials ##

#builder.explored_dataset = load_node(748569) ## Dataset to be passed to the committe evaluation
#builder.dataset = load_node(85953) ## Dataset selected to be labelled or already labelled (both labelled and unlabelled datasets are accepted in the same dataset)
#builder = models_from_trainingwc(builder, 1896245, get_labelled_dataset=True, get_config=True) ## populates builder with
builder = models_from_trainingwc(builder, 1906060, get_labelled_dataset=True, get_config=True) 
#builder = models_from_trainingwc(builder, 87443, get_labelled_dataset=True, get_config=True) ## populates builder with models (and eventually dataset and MACE parameters) from a previous training workflow
#builder.models_lammps = {"pot_1":load_node(85984), "pot_2":load_node(85995), "pot_3":load_node(86006), "pot_4":load_node(86017)} ## MACE potentials compiled for LAMMPS
#builder.models_ase = {"pot_1":load_node(85985), "pot_2":load_node(85996), "pot_3":load_node(86007), "pot_4":load_node(86018)} ## MACE potentials compiled for ASE

###############################################
# Thresholds on committe evaluation to select
# structures to be labelled
###############################################
builder.thr_energy          = Float(2e-3)
builder.thr_forces          = Float(5e-2)
builder.thr_stress          = Float(1e-2)
builder.max_selected_frames = Int(1000)


###############################################
# Setup dataset augmentation
###############################################

builder.dataset_augmentation.do_rattle_strain_defects           = Bool(True)
builder.dataset_augmentation.do_input                           = Bool(True)
builder.dataset_augmentation.do_isolated                        = Bool(True)
builder.dataset_augmentation.do_clusters                        = Bool(True)
builder.dataset_augmentation.do_slabs                           = Bool(True)
builder.dataset_augmentation.do_replication                     = Bool(True)
builder.dataset_augmentation.do_check_vacuum                    = Bool(True)
builder.dataset_augmentation.do_substitution                    = Bool(True)

builder.dataset_augmentation.rsd.params.rattle_fraction         = Float(0.6)
builder.dataset_augmentation.rsd.params.max_compressive_strain  = Float(0.3)
builder.dataset_augmentation.rsd.params.max_tensile_strain      = Float(0.3)
builder.dataset_augmentation.rsd.params.n_configs               = Int(8)
builder.dataset_augmentation.rsd.params.frac_vacancies          = Float(0.2)
builder.dataset_augmentation.rsd.params.vacancies_per_config    = Int(1)
builder.dataset_augmentation.clusters.n_clusters                = Int(8)
builder.dataset_augmentation.clusters.max_atoms                 = Int(3)
builder.dataset_augmentation.clusters.interatomic_distance      = Float(1.5)
builder.dataset_augmentation.slabs.miller_indices               = List([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]])
builder.dataset_augmentation.slabs.min_thickness                = Float(10)
builder.dataset_augmentation.slabs.max_atoms                    = Int(6)
builder.dataset_augmentation.replicate.min_dist                 = Float(24)
builder.dataset_augmentation.replicate.max_atoms                = Int(6)
builder.dataset_augmentation.vacuum                             = Float(10)
builder.dataset_augmentation.substitution.switches_fraction     = Float(0.2)
builder.dataset_augmentation.substitution.structures_fraction   = Float(0.1)

###############################################
# Setup Ab initio labelling
###############################################


builder.ab_initio_labelling.group_label                                                     = Str("graphene")
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.withmpi                     = True
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_wallclock_seconds       = QE_time
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.import_sys_environment      = False
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_memory_kb               = QE_mem
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.resources                   = {'num_machines': QE_machine["nodes"], 'num_mpiprocs_per_machine': QE_machine["taskpn"], 'num_cores_per_mpiproc': QE_machine['cpupt']}
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.account                     = QE_machine['account']
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.queue_name                  = QE_machine['partition']
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.custom_scheduler_commands   = f'#SBATCH --gres=gpu:{QE_machine["gpu"]} '
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.qos                         = QE_machine['qos']

### PW parameters are already populated once defining the builder according to pseudo_family and protocol
### Can be modified here if needed

# builder.ab_initio_labelling.quantumespresso.pw.pseudos = ...
# builder.ab_initio_labelling.quantumespresso.kpoints = KpointsData().set_kpoints_mesh([1, 1, 1])

# qe_parameters = builder.ab_initio_labelling.quantumespresso.pw.parameters.get_dict()
# print(qe_parameters)
# qe_parameters['ELECTRONS'] = {'conv_thr': Float(1.0e-8), 'mixing_beta': Float(0.5), 'mixing_mode': Str('local-TF'), }
# builder.ab_initio_labelling.quantumespresso.pw.parameters = Dict(qe_parameters)


###############################################
# Setup TRAINING
###############################################

builder.training.num_potentials = Int(3)


###############################################
# Setup MACE
###############################################

MACE_config = os.path.join(script_dir, 'mace_config.yml')
builder.training.mace.train.code = MACE_train_code
builder.training.mace.train.preprocess_code  = MACE_preprocess_code
builder.training.mace.train.postprocess_code = MACE_postprocess_code
builder.training.mace.train.do_preprocess = Bool(True)

with open(MACE_config, 'r') as yaml_file:
    mace_config = yaml.safe_load(yaml_file)
builder.training.mace.train.mace_config = Dict(mace_config)

builder.training.num_potentials = Int(5)
builder.training.mace.train.metadata.options.withmpi=False
builder.training.mace.train.metadata.options.resources = {
    'num_machines': MACE_machine['nodes'],
    'num_mpiprocs_per_machine': MACE_machine['taskpn'],
    'num_cores_per_mpiproc': MACE_machine['cpupt'],
}
builder.training.mace.train.metadata.options.max_wallclock_seconds = MACE_time
builder.training.mace.train.metadata.options.max_memory_kb = MACE_mem
builder.training.mace.train.metadata.options.import_sys_environment = False
builder.training.mace.train.metadata.options.account = MACE_machine['account']
builder.training.mace.train.metadata.options.queue_name = MACE_machine['partition']
builder.training.mace.train.metadata.options.qos = MACE_machine['qos']
builder.training.mace.train.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{MACE_machine['gpu']}"
builder.training.mace.train.metadata.options.prepend_text = """function mace_run_train(){
    srun mace_run_train $@
}
export -f mace_run_train"""


###############################################
# Setup META
###############################################

META_config                                                             = os.path.join(script_dir, 'meta_config.yml')
with open(META_config, 'r') as yaml_file:
    meta_config = yaml.safe_load(yaml_file)
builder.training.meta.train.meta_config                                 = Dict(meta_config)

builder.training.meta.train.code                                        = META_train_code
#builder.training.mace.train.preprocess_code                             = MACE_preprocess_code
#builder.training.mace.train.postprocess_code                            = MACE_postprocess_code
#builder.training.mace.train.do_preprocess                               = Bool(True)



builder.training.meta.train.metadata.options.withmpi                    = False
builder.training.meta.train.metadata.options.resources                  = {
    'num_machines': META_machine['nodes'],
    'num_mpiprocs_per_machine': META_machine['taskpn'],
    'num_cores_per_mpiproc': META_machine['cpupt'],
}
builder.training.meta.train.metadata.options.max_wallclock_seconds      = META_time
builder.training.meta.train.metadata.options.max_memory_kb              = META_mem
builder.training.meta.train.metadata.options.import_sys_environment     = False
builder.training.meta.train.metadata.options.account                    = META_machine['account']
builder.training.meta.train.metadata.options.queue_name                 = META_machine['partition']
builder.training.meta.train.metadata.options.qos                        = META_machine['qos']
builder.training.meta.train.metadata.options.custom_scheduler_commands  = f"#SBATCH --gres=gpu:{META_machine['gpu']}"
#builder.training.meta.train.metadata.options.prepend_text               = """function mace_run_train(){
#    srun mace_run_train $@
#}
#export -f mace_run_train""" ### This is needed to parallelize the training of MACE on multiple GPUs.



###############################################
# Setup LAMMPS
###############################################
builder.random_input_structures_lammps = Bool(False)
builder.num_random_structures_lammps = Int(1)
# builder.lammps_input_structures = load_node(933377)

# Generate the simple configuration of md parameters for LAMMPS
temperatures = [300]
pressures = [0]
steps = [100]
styles =  ["npt"]
timestep = 0.001
builder.exploration.params_list = generate_lammps_md_config(temperatures, pressures, steps, styles, timestep)
builder.exploration.parameters = Dict({'control':{'timestep': timestep,}, 'potential':{'neighbor_modify': ['one', '20000', 'page', '200000'], }})
builder.exploration.potential_pair_style = Str("metatomic")

#builder.exploration.md.lammps.settings = Dict({"additional_cmdline_params": ["-k", "on", "g", "1", "-sf", "kk"]})
builder.exploration.md.lammps.metadata.options.resources = {
    'num_machines': LAMMPS_machine['nodes'],
    'num_mpiprocs_per_machine': LAMMPS_machine['taskpn'],
    'num_cores_per_mpiproc': LAMMPS_machine['cpupt']
}
builder.exploration.md.lammps.metadata.options.max_wallclock_seconds = LAMMPS_time
builder.exploration.md.lammps.metadata.options.max_memory_kb = LAMMPS_mem
builder.exploration.md.lammps.metadata.options.import_sys_environment = False
builder.exploration.md.lammps.metadata.options.account = LAMMPS_machine['account']
builder.exploration.md.lammps.metadata.options.queue_name = LAMMPS_machine['partition']
builder.exploration.md.lammps.metadata.options.qos = LAMMPS_machine['qos']
builder.exploration.md.lammps.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{LAMMPS_machine['gpu']}"



builder.frame_extraction.sampling_time                                      = Float(0.2) # in ps how often frames are written to the trajectory file
builder.frame_extraction.thermalization_time                                = Float(0.0) # in ps how long the thermalization time is. Frames in that time are not considered



###############################################
# Setup committee Evaluation
###############################################

builder.committee_evaluation.code                                           = EVALUATION_code
builder.committee_evaluation.metadata.options.resources                     = {
    'num_machines': EVALUATION_machine['nodes'],
    'num_mpiprocs_per_machine': EVALUATION_machine['taskpn'],
    'num_cores_per_mpiproc': EVALUATION_machine['cpupt']
}
builder.committee_evaluation.metadata.options.max_wallclock_seconds         = EVALUATION_time
builder.committee_evaluation.metadata.options.max_memory_kb                 = EVALUATION_mem
builder.committee_evaluation.metadata.options.import_sys_environment        = False
builder.committee_evaluation.metadata.options.queue_name                    = EVALUATION_machine['partition']
builder.committee_evaluation.metadata.options.custom_scheduler_commands     = f"#SBATCH --gres=gpu:{EVALUATION_machine['gpu']}"
builder.committee_evaluation.metadata.options.qos                           = EVALUATION_machine['qos']
builder.committee_evaluation.metadata.options.account                       = EVALUATION_machine['account']
builder.committee_evaluation.metadata.computer                              = EVALUATION_computer 


calc = submit(builder)
print(f"Submitted calculation with PK = {calc.pk}")