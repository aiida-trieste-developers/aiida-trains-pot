from aiida.orm import load_code, load_node, load_group, Str, Dict, List, Group, Int, Data, Bool, Float, StructureData, FolderData
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.tools.groups import GroupPath
from pathlib import Path
from aiida.orm import PortableCode
from aiida.common.extendeddicts import AttributeDict
from aiida_lammps.data.potential import LammpsPotentialData
from ase.io import read
import yaml
import os
from pathlib import Path
load_profile()


KpointsData = DataFactory("core.array.kpoints")
NNIPWorkChain = WorkflowFactory('NNIPdevelopment.nnipdevelopment')


machine_dft = {
'time'                             : "00:05:00",
'nodes'                            : 1,
'mem'                              : "70GB",
'taskpn'                           : 1,
'taskps'                           : "1",
'cpupt'                            : "8",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'gpu'                              : "1",
'pool'                             : "1",
'poolx'                            : "1",
'pools'                            : "1",
'pooln'                            : "1",
'poolp'                            : "1",
'qos'                              : "boost_qos_dbg"
}


machine_mace = {
'time'                             : "00:01:00",
'nodes'                            : 1,
'mem'                              : "30GB",
'taskpn'                           : 1,
'taskps'                           : "1",
'cpupt'                            : "8",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'gpu'                              : "1",
'qos'                              : "boost_qos_dbg"
}

machine_lammps= {
'time'                             : "00:05:00",
'nodes'                            : 1,
'mem'                              : "30GB",
'taskpn'                           : 1,
'taskps'                           : "1",
'cpupt'                            : "8",
'account'                          : "CNHPC_1491920",
'partition'                        : "boost_usr_prod",
'gpu'                              : "1",
'qos'                              : "boost_qos_dbg"
}

#machine_evaluation = {
#'time'                             : "00:30:00",
#'nodes'                            : 1,
#'mem'                              : "7GB",
#'taskpn'                           : 1,
#'taskps'                           : "1",
#'cpupt'                            : "1",
#'account'                          : "",
#'partition'                        : "main",
#'gpu'                              : "a30",
#'pool'                             : "1"
#}

machine_evaluation = {
 'time'                             : "00:05:00",
 'nodes'                            : 1,
 'mem'                              : "30GB",
 'taskpn'                           : 1,
 'taskps'                           : "1",
 'cpupt'                            : "8",
 'account'                          : "CNHPC_1491920",
 'partition'                        : "boost_usr_prod",
 'gpu'                              : "1",
 'qos'                              : "boost_qos_dbg"
 }


description = "mote"
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

mem_dft = get_memory(machine_dft['mem'])
time_dft = get_time(machine_dft['time'])

mem_mace = get_memory(machine_mace['mem'])
time_mace = get_time(machine_mace['time'])

mem_lammps = get_memory(machine_lammps['mem'])
time_lammps = get_time(machine_lammps['time'])

mem_evaluation = get_memory(machine_evaluation['mem'])
time_evaluation = get_time(machine_evaluation['time'])


# structures = [load_node(25538), load_node(25536)] #wte2
#structures = [StructureData(ase=read('/home/bidoggia/onedrive/aiida/git/NNIPDevelopment/examples/gr8x8.xyz'))]
structures = [StructureData(ase=read('/home/nataliia/Documents/aiida_scripts/examples/gr8x8.xyz'))]

kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])
pseudo_family = load_group('SSSP/1.3/PBE/precision')
cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structures[0], unit='Ry')





#builder = NNIPWorkChain.get_builder_from_protocol(structures, qe_code = load_code('qe7.2-pw@leo1_scratch_bind'))
builder = NNIPWorkChain.get_builder_from_protocol(structures, qe_code = load_code('qe7.2-pw@leo2_scratch_bind'))
builder.do_data_generation = Bool(False)
builder.do_dft = Bool(False)
builder.do_mace = Bool(True)
builder.do_md = Bool(True)
builder.max_loops = Int(2)
#builder.labelled_list = load_node(56495)
builder.labelled_list = load_node(93634)
#builder.mace_lammps_potentials = {"pot_1":load_node(56510),"pot_2":load_node(56532), "pot_1":load_node(56543),"pot_2":load_node(56521)}
#builder.mace_ase_potentials = {"pot_1":load_node(56511),"pot_2":load_node(56533), "pot_1":load_node(56544),"pot_2":load_node(56522)}

builder.thr_energy = Float(1e-3)
builder.thr_forces = Float(1e-1)
builder.thr_stress = Float(1e-1)

builder.datagen.do_rattle = Bool(True)
builder.datagen.do_input = Bool(True)
builder.datagen.do_isolated = Bool(True)
builder.datagen.rattle.params.rattle_fraction = Float(0.1)
builder.datagen.rattle.params.max_sigma_strain = Float(0.1)
builder.datagen.rattle.params.n_configs = Int(20)
builder.datagen.rattle.params.frac_vacancies = Float(0.1)
builder.datagen.rattle.params.vacancies_per_config = Int(1)


builder.dft.pw.metadata.description = 'test'
builder.dft.pw.metadata.options.withmpi=True
builder.dft.pw.metadata.options.max_wallclock_seconds = time_dft
builder.dft.pw.metadata.options.import_sys_environment = False
builder.dft.pw.metadata.options.max_memory_kb = mem_dft
builder.dft.pw.metadata.options.resources = {'num_machines': machine_dft["nodes"], 'num_mpiprocs_per_machine': machine_dft["taskpn"], 'num_cores_per_mpiproc': machine_dft['cpupt']}
builder.dft.pw.metadata.options.account = machine_dft['account']
builder.dft.pw.metadata.options.queue_name = machine_dft['partition']
builder.dft.pw.metadata.options.custom_scheduler_commands = f'#SBATCH --gres=gpu:{machine_dft["gpu"]} '
builder.dft.pw.metadata.options.qos = machine_dft['qos']
builder.dft.pw.pseudos = pseudo_family.get_pseudos(structure=structures[0])
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

#builder.mace.mace.code = load_code('mace_pub@leo1_scratch')
builder.mace.mace.code = load_code('mace13@leo1_scratch_bind')
#with open('/home/bidoggia/onedrive/aiida/git/NNIPDevelopment/examples/mace_config.yml', 'r') as yaml_file:
with open('examples/mace_config.yml', 'r') as yaml_file:
    mace_config = yaml.safe_load(yaml_file)
builder.mace.mace.mace_config = Dict(mace_config)
# Save the checkpoints folder as FolderData
# folder_path = 'checkpoints'  # Replace with the actual path to your checkpoints folder
# checkpoints_folder_data = FolderData()
# for root, _, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             relative_path = os.path.relpath(file_path, folder_path)
#             with open(file_path, 'rb') as handle:
#                 checkpoints_folder_data.put_object_from_filelike(handle, relative_path)
# builder.mace.checkpoints = checkpoints_folder_data
builder.mace.num_potentials = Int(4)
builder.mace.mace.metadata.options.resources = {
    'num_machines': machine_mace['nodes'],
    'num_mpiprocs_per_machine': machine_mace['taskpn'],
    'num_cores_per_mpiproc': machine_mace['cpupt'],
  #  'num_gpus_per_machine': machine_mace['gpu'],
}
builder.mace.mace.metadata.options.max_wallclock_seconds = time_mace
builder.mace.mace.metadata.options.max_memory_kb = mem_mace
builder.mace.mace.metadata.options.import_sys_environment = False
builder.mace.mace.metadata.options.account = machine_mace['account']
builder.mace.mace.metadata.options.queue_name = machine_mace['partition']
builder.mace.mace.metadata.options.qos = machine_mace['qos']
builder.mace.mace.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{machine_mace['gpu']}"

code = PortableCode(
    label='preprocess',
    filepath_files=Path('/home/nataliia/Documents/aiida_scripts/src/NNIPdevelopment/mace/mace_train_wc/'),
    filepath_executable='preprocess_config.py'
)
code.store()

#builder.md.code = load_code('lmp4mace2@leo1_scratch')
#builder.md.code = load_code('lmp4mace3@leo1_scratch_bind')
builder.md.lammps.code = load_code('lmp4mace_new@leo1_scratch_bind')
#builder.md.temperatures = List([30, 50])
#builder.md.pressures = List([0])
#builder.md.num_steps = Int(500)
#builder.md.dt = Float(0.00242)
builder.md.parent_folder = Str(Path(__file__).resolve().parent)
# Parameters to control the input file generation
_parameters = AttributeDict()
# Control section specifying global simulation parameters
_parameters.control = AttributeDict()
# Types of units to be used in the calculation
_parameters.control.units = "metal"
# Size of the time step in the units previously defined
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
_parameters.md = {
    "integration": {
        "style": "npt",
        "constraints": {
            "temp": [30, 30, 0.242],
            "x": [0.0, 0.0, 2.42],
            "y": [0.0, 0.0, 2.42],
        },
    },
    "max_number_steps": 500,
    "velocity": [{"create": {"temp": 30, "seed": 633}, "group": "all"}],
}
# Control how often the computes are printed to file
#_parameters.dump = {"dump_rate": 1000}
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
# (THIS DOES NOT STORE IT IN THE DATABASE JUST PRINTS IT)
_parameters.restart = {"print_final": True}
# Convert the parameters to an AiiDA data structure
PARAMETERS = orm.Dict(dict=_parameters)

# Controlling parameters on how the LAMMPS calculation is performed
_settings = AttributeDict()
# Whether or not to store the restart file in the database
_settings.store_restart = True
_settings.additional_cmdline_params = ["-k", "on", "g", "1", "-sf", "kk"]

# Store the setting parameters in an AiiDA datastructure
SETTINGS = orm.Dict(dict=_settings)#
builder.md.lammps.settings = SETTINGS
builder.md.lammps.parameters = PARAMETERS#
builder.md.lammps.metadata.options.resources = {
    'num_machines': machine_lammps['nodes'],
    'num_mpiprocs_per_machine': machine_lammps['taskpn'],
    'num_cores_per_mpiproc': machine_lammps['cpupt']
}
builder.md.lammps.metadata.options.max_wallclock_seconds = time_lammps
builder.md.lammps.metadata.options.max_memory_kb = mem_lammps
builder.md.lammps.metadata.options.import_sys_environment = False
builder.md.lammps.metadata.options.account = machine_lammps['account']
builder.md.lammps.metadata.options.queue_name = machine_lammps['partition']
builder.md.lammps.metadata.options.qos = machine_lammps['qos']
builder.md.lammps.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{machine_lammps['gpu']}"

builder.frame_extraction.correlation_time = Float(0.242)
builder.frame_extraction.thermalization_time = Float(2.42)

# builder.cometee_evaluation.code = load_code('cometee-evaluation@leo1_scratch')
#builder.cometee_evaluation.code = load_code('cometee-evaluation@bora')
builder.cometee_evaluation.code = load_code('cometee_evaluation@leo1_scratch_bind')
builder.cometee_evaluation.metadata.options.resources = {
    'num_machines': machine_evaluation['nodes'],
    'num_mpiprocs_per_machine': machine_evaluation['taskpn'],
    'num_cores_per_mpiproc': machine_evaluation['cpupt']
}
builder.cometee_evaluation.metadata.options.max_wallclock_seconds = time_evaluation
builder.cometee_evaluation.metadata.options.max_memory_kb = mem_evaluation
builder.cometee_evaluation.metadata.options.import_sys_environment = False
builder.cometee_evaluation.metadata.options.queue_name = machine_evaluation['partition']
builder.cometee_evaluation.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{machine_evaluation['gpu']}"
builder.cometee_evaluation.metadata.options.qos = machine_evaluation['qos']
builder.cometee_evaluation.metadata.options.account = machine_evaluation['account']




submit(builder) #.id