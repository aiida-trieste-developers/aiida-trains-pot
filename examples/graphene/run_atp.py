from aiida.orm import load_code, load_node, load_group, load_computer, Str, Dict, List, Int, Bool, Float, StructureData
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory, DataFactory
from pathlib import Path
from aiida.common.extendeddicts import AttributeDict
from ase.io import read
import yaml
import os
load_profile()

KpointsData = DataFactory("core.array.kpoints")
TrainsPot   = WorkflowFactory('trains_pot.workflow')

####################################################################
#                     START MACHINE PARAMETERS                     #
####################################################################


QE_code                 = load_code('qe7.2-pw@leo1_scratch_bind')
MACE_train_code         = load_code('mace_train@leo1_scratch')
MACE_preprocess_code    = load_code('mace_preprocess@leo1_scratch')
MACE_postprocess_code   = load_code('mace_postprocess@leo1_scratch')
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
'time'                             : "00:01:00",
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
builder.do_data_generation = Bool(True)
builder.do_dft = Bool(True)
builder.do_mace = Bool(True)
builder.do_md = Bool(True)
builder.max_loops = Int(2)
#builder.labelled_list = load_node(113290)
#builder.mace_lammps_potentials = {"pot_1":load_node(113311),"pot_2":load_node(113321),"pot_3":load_node(113331),"pot_4":load_node(113341)}
#builder.mace_ase_potentials = {"pot_1":load_node(113312),"pot_2":load_node(113322),"pot_3":load_node(113332),"pot_4":load_node(113342)}

builder.thr_energy = Float(1e-3)
builder.thr_forces = Float(1e-1)
builder.thr_stress = Float(1e-1)


###############################################
# Setup Datageneration
###############################################

builder.datagen.do_rattle = Bool(True)
builder.datagen.do_input = Bool(True)
builder.datagen.do_isolated = Bool(True)
builder.datagen.rattle.params.rattle_fraction = Float(0.1)
builder.datagen.rattle.params.max_sigma_strain = Float(0.1)
builder.datagen.rattle.params.n_configs = Int(20)
builder.datagen.rattle.params.frac_vacancies = Float(0.1)
builder.datagen.rattle.params.vacancies_per_config = Int(1)



###############################################
# Setup Quantum ESPRESSO
###############################################

kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])
pseudo_family = load_group('SSSP/1.3/PBE/precision')
cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=input_structures[0], unit='Ry')

builder.dft.pw.code = QE_code
builder.dft.pw.metadata.options.withmpi=True
builder.dft.pw.metadata.options.max_wallclock_seconds = QE_time
builder.dft.pw.metadata.options.import_sys_environment = False
builder.dft.pw.metadata.options.max_memory_kb = QE_mem
builder.dft.pw.metadata.options.resources = {'num_machines': QE_machine["nodes"], 'num_mpiprocs_per_machine': QE_machine["taskpn"], 'num_cores_per_mpiproc': QE_machine['cpupt']}
builder.dft.pw.metadata.options.account = QE_machine['account']
builder.dft.pw.metadata.options.queue_name = QE_machine['partition']
builder.dft.pw.metadata.options.custom_scheduler_commands = f'#SBATCH --gres=gpu:{QE_machine["gpu"]} '
builder.dft.pw.metadata.options.qos = QE_machine['qos']
builder.dft.pw.pseudos = pseudo_family.get_pseudos(structure=input_structures[0])
builder.dft.kpoints = kpoints
builder.dft.pw.parameters = Dict({'SYSTEM':
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
builder.mace.mace.code = MACE_train_code
builder.mace.mace.preprocess_code  = MACE_preprocess_code
builder.mace.mace.postprocess_code = MACE_postprocess_code
# builder.mace.mace.do_preprocess = Bool(True)

with open(MACE_config, 'r') as yaml_file:
    mace_config = yaml.safe_load(yaml_file)
builder.mace.mace.mace_config = Dict(mace_config)

builder.mace.num_potentials = Int(4)
builder.mace.mace.metadata.options.resources = {
    'num_machines': MACE_machine['nodes'],
    'num_mpiprocs_per_machine': MACE_machine['taskpn'],
    'num_cores_per_mpiproc': MACE_machine['cpupt'],
}
builder.mace.mace.metadata.options.max_wallclock_seconds = MACE_time
builder.mace.mace.metadata.options.max_memory_kb = MACE_mem
builder.mace.mace.metadata.options.import_sys_environment = False
builder.mace.mace.metadata.options.account = MACE_machine['account']
builder.mace.mace.metadata.options.queue_name = MACE_machine['partition']
builder.mace.mace.metadata.options.qos = MACE_machine['qos']
builder.mace.mace.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{MACE_machine['gpu']}"



###############################################
# Setup LAMMPS
###############################################

builder.md.lammps.code = LAMMPS_code
md_params_yaml = os.path.join(script_dir, 'lammps_md_params.yml')
with open(md_params_yaml, 'r') as yaml_file:
    md_params_list = yaml.safe_load(yaml_file)
builder.md.md_params_list = List(md_params_list)
#builder.md.temperatures = List([30, 50])
#builder.md.pressures = List([0])
#builder.md.num_steps = Int(500)
#builder.md.dt = Float(0.00242)
builder.md.parent_folder = Str(Path(__file__).resolve().parent)
_parameters = AttributeDict()
_parameters.control = AttributeDict()
_parameters.control.units = "metal"
_parameters.control.timestep = 0.00242
_parameters.control.newton = "on"
# Set of computes to be evaluated during the calculation
#_parameters.compute = {
#    "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
#    "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
#    "stress/atom": [{"type": ["NULL"], "group": "all"}],
#    "pressure": [{"type": ["thermo_temp"], "group": "all"}],
#}
# Set of values to control the behaviour of the molecular dynamics calculation
#_parameters.md = {
#    "integration": {
#        "style": "npt",
#        "constraints": {
#            "temp": [30, 30, 0.242],
#            "x": [0.0, 0.0, 2.42],
#            "y": [0.0, 0.0, 2.42],
#        },
#    },
#    "max_number_steps": 100,
#    "velocity": [{"create": {"temp": 30, "seed": 633}, "group": "all"}],
#}
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
builder.md.lammps.settings = SETTINGS
builder.md.parameters = PARAMETERS
builder.md.lammps.metadata.options.resources = {
    'num_machines': LAMMPS_machine['nodes'],
    'num_mpiprocs_per_machine': LAMMPS_machine['taskpn'],
    'num_cores_per_mpiproc': LAMMPS_machine['cpupt']
}
builder.md.lammps.metadata.options.max_wallclock_seconds = LAMMPS_time
builder.md.lammps.metadata.options.max_memory_kb = LAMMPS_mem
builder.md.lammps.metadata.options.import_sys_environment = False
builder.md.lammps.metadata.options.account = LAMMPS_machine['account']
builder.md.lammps.metadata.options.queue_name = LAMMPS_machine['partition']
builder.md.lammps.metadata.options.qos = LAMMPS_machine['qos']
builder.md.lammps.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{LAMMPS_machine['gpu']}"

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




calc = submit(builder)
print(f"Submitted calculation with PK = {calc.pk}")