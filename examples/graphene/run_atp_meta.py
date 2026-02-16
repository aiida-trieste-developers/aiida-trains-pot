"""Example script to run the AiiDA TrainsPot workflow with META training on graphene."""

import os

import yaml

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Bool, Dict, Float, Int, List, Str, load_code, load_computer
from aiida.plugins import DataFactory
from ase.io import read


from aiida_trains_pot.data.pesdata import PESData
from aiida_trains_pot.utils.generate_config import generate_lammps_md_config
from aiida_trains_pot.aiida_trains_pot_workflow.aiida_trains_pot_workflow import TrainsPotWorkChain

load_profile()

KpointsData = DataFactory("core.array.kpoints")

####################################################################
#                     START MACHINE PARAMETERS                     #
####################################################################


QE_code = load_code("<code>@<computer>")
MACE_train_code = load_code("<code>@<computer>")
MACE_preprocess_code = load_code("<code>@<computer>")
MACE_postprocess_code = load_code("<code>@<computer>")
META_train_code = load_code("<code>@<computer>")
LAMMPS_code = load_code("<code>@<computer>")
EVALUATION_code = load_code("<code>")
EVALUATION_computer = load_computer("<computer>")

HPC_account = "<account>"

QE_machine = {
    "time": "00:05:00",
    "nodes": 1,
    "gpu": "1",
    "taskpn": 1,
    "cpupt": "8",
    "mem": "70GB",
    "account": HPC_account,
    "partition": "boost_usr_prod",
    "qos": "normal",
}

MACE_machine = {
    "time": "00:30:00",
    "nodes": 1,
    "gpu": "1",
    "taskpn": 1,
    "cpupt": "8",
    "mem": "30GB",
    "account": HPC_account,
    "partition": "boost_usr_prod",
    "qos": "normal",
}

META_machine = {
    "time": "00:05:00",
    "nodes": 1,
    "gpu": "1",
    "taskpn": 1,
    "cpupt": "8",
    "mem": "30GB",
    "account": HPC_account,
    "partition": "boost_usr_prod",
    "qos": "normal",
}

LAMMPS_machine = {
    "time": "00:30:00",
    "nodes": 1,
    "gpu": "1",
    "taskpn": 1,
    "cpupt": "8",
    "mem": "30GB",
    "account": HPC_account,
    "partition": "boost_usr_prod",
    "qos": "normal",
}

EVALUATION_machine = {
    "time": "00:30:00",
    "nodes": 1,
    "gpu": "1",
    "taskpn": 1,
    "cpupt": "8",
    "mem": "30GB",
    "account": HPC_account,
    "partition": "boost_usr_prod",
    "qos": "normal",
}

####################################################################
#                      END MACHINE PARAMETERS                      #
####################################################################


def get_memory(mem):
    """Convert memory in MB, GB, KB format to KB."""
    if mem.find("MB") != -1:
        mem = int(mem.replace("MB", "")) * 1024
    elif mem.find("GB") != -1:
        mem = int(mem.replace("GB", "")) * 1024 * 1024
    elif mem.find("KB") != -1:
        mem = int(mem.replace("KB", ""))
    return mem


def get_time(time):
    """Convert time in HH:MM:SS format to seconds."""
    time = time.split(":")
    time_sec = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    return time_sec


QE_mem = get_memory(QE_machine["mem"])
QE_time = get_time(QE_machine["time"])

MACE_mem = get_memory(MACE_machine["mem"])
MACE_time = get_time(MACE_machine["time"])

META_mem = get_memory(META_machine["mem"])
META_time = get_time(META_machine["time"])

LAMMPS_mem = get_memory(LAMMPS_machine["mem"])
LAMMPS_time = get_time(LAMMPS_machine["time"])

EVALUATION_mem = get_memory(EVALUATION_machine["mem"])
EVALUATION_time = get_time(EVALUATION_machine["time"])

script_dir = os.path.dirname(os.path.abspath(__file__))

###############################################
# Input structures
###############################################

input_structures = PESData([read(os.path.join(script_dir, "gr8x8.xyz"))])

###############################################
# Setup TrainsPot workflow
###############################################

builder = TrainsPotWorkChain.get_builder(
    abinitiolabeling_code=QE_code,
    abinitiolabeling_protocol="fast",
    pseudo_family="SSSP/1.3/PBE/efficiency",
    md_code=LAMMPS_code,
    # md_protocol               = 'vdw_d2',
    dataset=input_structures,
)
builder.do_dataset_augmentation = Bool(False)
builder.do_ab_initio_labelling = Bool(False)
builder.training_engine = Str("META")
builder.do_training = Bool(True)
builder.do_exploration = Bool(True)
builder.max_loops = Int(2)

## Additional inputs for restart from previous runs or to start with a previous dataset ##

## Dataset to be passed to the committe evaluation
# builder.explored_dataset = load_node(<node_uuid>)

## Dataset selected to be labelled or already labelled
## (both labelled and unlabelled datasets are accepted in the same dataset)
# builder.dataset = load_node(<node_uuid>)

###############################################
# Thresholds on committee evaluation to select
# structures to be labelled
###############################################
builder.thr_energy = Float(2e-3)
builder.thr_forces = Float(5e-2)
builder.thr_stress = Float(1e-2)
builder.max_selected_frames = Int(10)


###############################################
# Setup dataset augmentation
###############################################

builder.dataset_augmentation.do_rattle_strain_defects = Bool(True)
builder.dataset_augmentation.do_input = Bool(True)
builder.dataset_augmentation.do_isolated = Bool(True)
builder.dataset_augmentation.do_clusters = Bool(True)
builder.dataset_augmentation.do_slabs = Bool(True)
builder.dataset_augmentation.do_replication = Bool(True)
builder.dataset_augmentation.do_check_vacuum = Bool(True)
builder.dataset_augmentation.do_substitution = Bool(True)
builder.dataset_augmentation.do_alloys = Bool(False)

builder.dataset_augmentation.rsd.params.rattle_fraction = Float(0.6)
builder.dataset_augmentation.rsd.params.max_compressive_strain = Float(0.3)
builder.dataset_augmentation.rsd.params.max_tensile_strain = Float(0.3)
builder.dataset_augmentation.rsd.params.n_configs = Int(8)
builder.dataset_augmentation.rsd.params.frac_vacancies = Float(0.2)
builder.dataset_augmentation.rsd.params.vacancies_per_config = Int(1)
builder.dataset_augmentation.clusters.n_clusters = Int(8)
builder.dataset_augmentation.clusters.max_atoms = Int(3)
builder.dataset_augmentation.clusters.interatomic_distance = Float(1.5)
builder.dataset_augmentation.slabs.miller_indices = List(
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]]
)
builder.dataset_augmentation.slabs.min_thickness = Float(10)
builder.dataset_augmentation.slabs.max_atoms = Int(6)
builder.dataset_augmentation.replicate.min_dist = Float(24)
builder.dataset_augmentation.replicate.max_atoms = Int(6)
builder.dataset_augmentation.vacuum = Float(10)
builder.dataset_augmentation.substitution.switches_fraction = Float(0.2)
builder.dataset_augmentation.substitution.structures_fraction = Float(0.1)
builder.dataset_augmentation.alloys.fixed_species = List([])
builder.dataset_augmentation.alloys.alloy_species = List(["B", "N"])
builder.dataset_augmentation.alloys.num_structures = Int(8)
builder.dataset_augmentation.alloys.fractions = List([0.1, 0.5, 0.9])

###############################################
# Setup Ab initio labelling
###############################################


builder.ab_initio_labelling.group_label = Str("graphene")
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.withmpi = True
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_wallclock_seconds = QE_time
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.import_sys_environment = False
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_memory_kb = QE_mem
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.resources = {
    "num_machines": QE_machine["nodes"],
    "num_mpiprocs_per_machine": QE_machine["taskpn"],
    "num_cores_per_mpiproc": QE_machine["cpupt"],
}
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.account = QE_machine["account"]
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.queue_name = QE_machine["partition"]
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.custom_scheduler_commands = (
    f'#SBATCH --gres=gpu:{QE_machine["gpu"]} '
)
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.qos = QE_machine["qos"]

qe_parameters = builder.ab_initio_labelling.quantumespresso.pw.parameters.get_dict()
qe_parameters["ELECTRONS"] = {
    "conv_thr": Float(1.0e-8),
    "mixing_beta": Float(0.5),
    "mixing_mode": Str("local-TF"),
}

qe_parameters["SYSTEM"] = {
    "degauss": 0.0125,
    "occupations": "smearing",
    "smearing": "cold",
}

builder.ab_initio_labelling.quantumespresso.pw.parameters = Dict(qe_parameters)


###############################################
# Setup TRAINING
###############################################

builder.training.num_potentials = Int(3)

###############################################
# Setup META
###############################################

META_config = os.path.join(script_dir, "meta_config.yml")
with open(META_config) as yaml_file:
    meta_config = yaml.safe_load(yaml_file)
builder.training.meta.train.meta_config = Dict(meta_config)

builder.training.meta.train.code = META_train_code

builder.training.meta.train.metadata.options.withmpi = False
builder.training.meta.train.metadata.options.resources = {
    "num_machines": META_machine["nodes"],
    "num_mpiprocs_per_machine": META_machine["taskpn"],
    "num_cores_per_mpiproc": META_machine["cpupt"],
}
builder.training.meta.train.metadata.options.max_wallclock_seconds = META_time
builder.training.meta.train.metadata.options.max_memory_kb = META_mem
builder.training.meta.train.metadata.options.import_sys_environment = False
builder.training.meta.train.metadata.options.account = META_machine["account"]
builder.training.meta.train.metadata.options.queue_name = META_machine["partition"]
builder.training.meta.train.metadata.options.qos = META_machine["qos"]
builder.training.meta.train.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{META_machine['gpu']}"


###############################################
# Setup LAMMPS
###############################################
builder.random_input_structures_lammps = Bool(False)
builder.num_random_structures_lammps = Int(5)
## PESData node containing structures to be used as input in MD
# builder.lammps_input_structures = load_node(<node_uuid>)

# Generate the simple configuration of md parameters for LAMMPS
temperatures = [3000]
pressures = [0]
steps = [100]
styles = ["npt"]
timestep = 0.001
builder.exploration.params_list = generate_lammps_md_config(temperatures, pressures, steps, styles, timestep)
builder.exploration.parameters = Dict(
    {
        "control": {
            "timestep": timestep,
        },
        "potential": {
            "neighbor_modify": ["one", "20000", "page", "200000"],
        },
    }
)
builder.exploration.potential_pair_style = Str("metatomic")

# builder.exploration.md.lammps.settings = Dict({"additional_cmdline_params": ["-k", "on", "g", "1", "-sf", "kk"]})
builder.exploration.md.lammps.metadata.options.resources = {
    "num_machines": LAMMPS_machine["nodes"],
    "num_mpiprocs_per_machine": LAMMPS_machine["taskpn"],
    "num_cores_per_mpiproc": LAMMPS_machine["cpupt"],
}
builder.exploration.md.lammps.metadata.options.max_wallclock_seconds = LAMMPS_time
builder.exploration.md.lammps.metadata.options.max_memory_kb = LAMMPS_mem
builder.exploration.md.lammps.metadata.options.import_sys_environment = False
builder.exploration.md.lammps.metadata.options.account = LAMMPS_machine["account"]
builder.exploration.md.lammps.metadata.options.queue_name = LAMMPS_machine["partition"]
builder.exploration.md.lammps.metadata.options.qos = LAMMPS_machine["qos"]
builder.exploration.md.lammps.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{LAMMPS_machine['gpu']}"

# How often to extract frames from the MD trajectory (in LAMMPS time units)
builder.frame_extraction.sampling_time = Float(0.02)
# Thermalization time (in LAMMPS units). Frames in thermalization_time are not considered
builder.frame_extraction.thermalization_time = Float(0.0)


###############################################
# Setup committee Evaluation
###############################################

builder.committee_evaluation.code = EVALUATION_code
builder.committee_evaluation.metadata.options.resources = {
    "num_machines": EVALUATION_machine["nodes"],
    "num_mpiprocs_per_machine": EVALUATION_machine["taskpn"],
    "num_cores_per_mpiproc": EVALUATION_machine["cpupt"],
}
builder.committee_evaluation.metadata.options.max_wallclock_seconds = EVALUATION_time
builder.committee_evaluation.metadata.options.max_memory_kb = EVALUATION_mem
builder.committee_evaluation.metadata.options.import_sys_environment = False
builder.committee_evaluation.metadata.options.queue_name = EVALUATION_machine["partition"]
builder.committee_evaluation.metadata.options.custom_scheduler_commands = (
    f"#SBATCH --gres=gpu:{EVALUATION_machine['gpu']}"
)
builder.committee_evaluation.metadata.options.qos = EVALUATION_machine["qos"]
builder.committee_evaluation.metadata.options.account = EVALUATION_machine["account"]
builder.committee_evaluation.metadata.computer = EVALUATION_computer


calc = submit(builder)
print(f"Submitted calculation with PK = {calc.pk}")
