from aiida.orm import load_code, load_node, load_group, load_computer, Str, Dict, List, Int, Bool, Float, StructureData
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory, DataFactory
from pathlib import Path
from aiida.common.extendeddicts import AttributeDict
from ase.io import read
import yaml
import os
from aiida_trains_pot.utils.restart import models_from_trainingwc
from aiida_trains_pot.utils.generate_config import generate_lammps_md_config
load_profile()

PESData = DataFactory('pesdata')
KpointsData = DataFactory("core.array.kpoints")
TrainsPot   = WorkflowFactory('trains_pot.workflow')

####################################################################
#                     START MACHINE PARAMETERS                     #
####################################################################


QE_code                 = load_code('qe7.2-pw@leo1_scratch_bind')
MACE_train_code         = load_code('mace_train@leo1_scratch_mace')
MACE_preprocess_code    = load_code('mace_preprocess@leo1_scratch_mace')
MACE_postprocess_code   = load_code('mace_postprocess@leo1_scratch_mace')
LAMMPS_code             = load_code('lmp4mace@leo1_scratch')
EVALUATION_code         = load_code('committee_evaluation_portable')

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

EVALUATION_machine = {
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

LAMMPS_mem = get_memory(LAMMPS_machine['mem'])
LAMMPS_time = get_time(LAMMPS_machine['time'])

EVALUATION_mem = get_memory(EVALUATION_machine['mem'])
EVALUATION_time = get_time(EVALUATION_machine['time'])

script_dir = os.path.dirname(os.path.abspath(__file__))

###############################################
# Input structures
###############################################

input_structures = [StructureData(ase=read(os.path.join(script_dir, 'gr8x8.xyz')))]


###############################################
# Setup TrainsPot worflow
###############################################

builder = TrainsPot.get_builder()
builder.structures =  {f'structure_{i}':input_structures[i] for i in range(len(input_structures))}
# builder = TrainsPot.get_builder_from_protocol(input_structures, qe_code = QE_code)
builder.do_dataset_augmentation = Bool(False)
builder.do_ab_initio_labelling = Bool(False)
builder.do_training = Bool(False)
builder.do_exploration = Bool(True)
builder.max_loops = Int(1)
# builder.explored_dataset = load_node(748569)
# builder.labelled_list = load_node(677593)
builder = models_from_trainingwc(builder, 87443, get_labelled_dataset=True, get_config=True)
#builder.dataset = load_node(85953)
#builder.models_lammps = {"pot_1":load_node(85984), "pot_2":load_node(85995), "pot_3":load_node(86006), "pot_4":load_node(86017)}
#builder.models_ase = {"pot_1":load_node(85985), "pot_2":load_node(85996), "pot_3":load_node(86007), "pot_4":load_node(86018)}

builder.thr_energy = Float(1e-3)
builder.thr_forces = Float(1e-1)
builder.thr_stress = Float(1e-1)


###############################################
# Setup dataset augmentation
###############################################

builder.dataset_augmentation.do_rattle = Bool(True)
builder.dataset_augmentation.do_input = Bool(True)
builder.dataset_augmentation.do_isolated = Bool(True)
builder.dataset_augmentation.rattle.params.rattle_fraction = Float(0.1)
builder.dataset_augmentation.rattle.params.max_sigma_strain = Float(0.1)
builder.dataset_augmentation.rattle.params.n_configs = Int(20)
builder.dataset_augmentation.rattle.params.frac_vacancies = Float(0.1)
builder.dataset_augmentation.rattle.params.vacancies_per_config = Int(1)

###############################################
# Setup Ab initio labelling
###############################################

kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])
pseudo_family = load_group('SSSP/1.3/PBE/precision')
cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=input_structures[0], unit='Ry')

builder.ab_initio_labelling.quantumespresso.pw.code = QE_code
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.withmpi=True
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_wallclock_seconds = QE_time
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.import_sys_environment = False
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_memory_kb = QE_mem
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.resources = {'num_machines': QE_machine["nodes"], 'num_mpiprocs_per_machine': QE_machine["taskpn"], 'num_cores_per_mpiproc': QE_machine['cpupt']}
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.account = QE_machine['account']
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.queue_name = QE_machine['partition']
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.custom_scheduler_commands = f'#SBATCH --gres=gpu:{QE_machine["gpu"]} '
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.qos = QE_machine['qos']
builder.ab_initio_labelling.quantumespresso.pw.pseudos = pseudo_family.get_pseudos(structure=input_structures[0])
builder.ab_initio_labelling.quantumespresso.kpoints = kpoints
builder.ab_initio_labelling.quantumespresso.pw.parameters = Dict({'SYSTEM':
                                                                {
                                                                    'ecutwfc': 10,#cutoff_wfc,
                                                                    'ecutrho': 40,#cutoff_rho,
                                                                    'degauss': 0.02,
                                                                    'occupations': 'smearing',
                                                                    'smearing': 'cold',
                                                                    'nosym': False,
                                                                },
                                                                'CONTROL': {'calculation': 'scf'},
                                                                'ELECTRONS':
                                                                    {
                                                                    'conv_thr': 1.0e-1,
                                                                    'mixing_beta': 0.5,
                                                                    'electron_maxstep': 50,
                                                                    'mixing_mode': 'local-TF',
                                                                    }
                                                                })



###############################################
# Setup MACE
###############################################

MACE_config = os.path.join(script_dir, 'mace_config.yml')
builder.training.mace.train.code = MACE_train_code
builder.training.mace.train.preprocess_code  = MACE_preprocess_code
builder.training.mace.train.postprocess_code = MACE_postprocess_code
# builder.training.mace.do_preprocess = Bool(True)

with open(MACE_config, 'r') as yaml_file:
    mace_config = yaml.safe_load(yaml_file)
builder.training.mace.train.mace_config = Dict(mace_config)

builder.training.num_potentials = Int(4)
builder.training.mace.train.metadata.options.withmpi=True
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



###############################################
# Setup LAMMPS
###############################################

builder.exploration.md.lammps.code = LAMMPS_code

# Read the configuration from file
#lammps_params_yaml = os.path.join(script_dir, 'lammps_md_params.yml')
#with open(lammps_params_yaml, 'r') as yaml_file:
#    lammps_params_list = yaml.safe_load(yaml_file)

# Generate the simple configuration
temperatures = [30, 35, 40, 45]
steps = [500] * len(temperatures)
styles =  ["npt"] * len(temperatures)  
constraints_template = {
    "temp": [30, 30, 0.242],
    "x": [0.0, 0.0, 2.42],
    "y": [0.0, 0.0, 2.42]
}
constraints = [constraints_template for _ in temperatures]
lammps_params_list = generate_lammps_md_config(temperatures, steps, constraints, styles)

builder.exploration.params_list = List(lammps_params_list)

builder.exploration.parent_folder = Str(Path(__file__).resolve().parent)
_parameters = AttributeDict()
_parameters.control = AttributeDict()
_parameters.control.units = "metal"
_parameters.control.timestep = 0.00242
_parameters.control.newton = "on"
_parameters.md = {}
_parameters.dump = {"dump_rate": 1} ## This parameter will be updated automatically based on the value of builder.frame_extraction.sampling_time
# Control how often the computes are printed to file
# Parameters used to pass special information about the structure
_parameters.structure = {"atom_style": "atomic", "atom_modify": "map yes", "boundary": "p p f"}
# Parameters controlling the global values written directly to the output
_parameters.thermo = {
    "printing_rate": 20,
    "thermo_printing": {
        "step": True,
        "pe": True,
        "ke": True,
        "press": True,
        "pxx": True,
        "pyy": True,
        "pzz": True,
    },
}
# Tell lammps to print the final restartfile
_parameters.restart = {"print_final": True}
PARAMETERS = Dict(dict=_parameters)

_settings = AttributeDict()
_settings.store_restart = True
_settings.additional_cmdline_params = ["-k", "on", "g", "1", "-sf", "kk"]

SETTINGS = Dict(dict=_settings)#
builder.exploration.md.lammps.settings = SETTINGS
builder.exploration.parameters = PARAMETERS
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

builder.frame_extraction.sampling_time = Float(0.242)
builder.frame_extraction.thermalization_time = Float(0)



###############################################
# Setup committee Evaluation
###############################################

builder.committee_evaluation.code = EVALUATION_code
builder.committee_evaluation.metadata.options.resources = {
    'num_machines': EVALUATION_machine['nodes'],
    'num_mpiprocs_per_machine': EVALUATION_machine['taskpn'],
    'num_cores_per_mpiproc': EVALUATION_machine['cpupt']
}
builder.committee_evaluation.metadata.options.max_wallclock_seconds = EVALUATION_time
builder.committee_evaluation.metadata.options.max_memory_kb = EVALUATION_mem
builder.committee_evaluation.metadata.options.import_sys_environment = False
builder.committee_evaluation.metadata.options.queue_name = EVALUATION_machine['partition']
builder.committee_evaluation.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{EVALUATION_machine['gpu']}"
builder.committee_evaluation.metadata.options.qos = EVALUATION_machine['qos']
builder.committee_evaluation.metadata.options.account = EVALUATION_machine['account']
builder.committee_evaluation.metadata.computer = load_computer('leo1_scratch')



type(builder)
calc = submit(builder)
print(f"Submitted calculation with PK = {calc.pk}")