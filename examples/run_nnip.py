from aiida.orm import load_code, load_node, load_group, Str, Dict, List, Group, Int, Data, Bool, Float, StructureData, FolderData
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.tools.groups import GroupPath
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
'account'                          : "IscrB_DeepVTe2",
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
'time'                             : "00:05:00",
'nodes'                            : 1,
'mem'                              : "30GB",
'taskpn'                           : 1,
'taskps'                           : "1",
'cpupt'                            : "8",
'account'                          : "IscrB_DeepVTe2",
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
'account'                          : "IscrB_DeepVTe2",
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
 'account'                          : "IscrB_DeepVTe2",
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
builder.max_loops = Int(3)
builder.labelled_list = load_node(44355)
#builder.labelled_list = load_node(74946)
# builder.mace_lammps_potential = load_node(47714)

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
    'num_cores_per_mpiproc': machine_mace['cpupt']
}
builder.mace.mace.metadata.options.max_wallclock_seconds = time_mace
builder.mace.mace.metadata.options.max_memory_kb = mem_mace
builder.mace.mace.metadata.options.import_sys_environment = False
builder.mace.mace.metadata.options.account = machine_mace['account']
builder.mace.mace.metadata.options.queue_name = machine_mace['partition']
builder.mace.mace.metadata.options.qos = machine_mace['qos']
builder.mace.mace.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{machine_mace['gpu']}"


#builder.md.code = load_code('lmp4mace2@leo1_scratch')
builder.md.code = load_code('lmp4mace@leo2_scratch_bind')
builder.md.temperatures = List([30, 50])
builder.md.pressures = List([0])
builder.md.num_steps = Int(500)
builder.md.dt = Float(0.00242)
builder.md.parent_folder = Str(Path(__file__).resolve().parent)
builder.md.lmp.metadata.options.resources = {
    'num_machines': machine_lammps['nodes'],
    'num_mpiprocs_per_machine': machine_lammps['taskpn'],
    'num_cores_per_mpiproc': machine_lammps['cpupt']
}
builder.md.lmp.metadata.options.max_wallclock_seconds = time_lammps
builder.md.lmp.metadata.options.max_memory_kb = mem_lammps
builder.md.lmp.metadata.options.import_sys_environment = False
builder.md.lmp.metadata.options.account = machine_lammps['account']
builder.md.lmp.metadata.options.queue_name = machine_lammps['partition']
builder.md.lmp.metadata.options.qos = machine_lammps['qos']
builder.md.lmp.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{machine_lammps['gpu']}"

builder.frame_extraction.correlation_time = Float(0.242)
builder.frame_extraction.thermalization_time = Float(2.42)

# builder.cometee_evaluation.code = load_code('cometee-evaluation@leo1_scratch')
#builder.cometee_evaluation.code = load_code('cometee-evaluation@bora')
builder.cometee_evaluation.code = load_code('cometee-evaluation2@leo1_scratch_bind')
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
# builder.cometee_evaluation.metadata.options.qos = machine_evaluation['qos']
# builder.cometee_evaluation.metadata.options.account = machine_evaluation['account']




submit(builder) #.id