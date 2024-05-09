# -*- coding: utf-8 -*-
"""Launch a calculation using the 'diff-tutorial' plugin"""
from pathlib import Path
import aiida
from aiida.plugins import WorkflowFactory, DataFactory
from ase.io import read
from aiida.engine import submit, run
from aiida.orm import load_group, load_node, Float, load_code, Int, Str,StructureData, SinglefileData, Dict
import numpy as np
from LoadStructure.pdb_loader import load_structures_from_folder
#from RuttleStructure.RuttleStructureWorkChain import RuttleStructureWorkChain


LammpsWorkChain = WorkflowFactory('lammpsworkchain')
KpointsData = DataFactory("core.array.kpoints")
QECalculationWorkChain = WorkflowFactory('qecalculation')
MaceWorkChain = WorkflowFactory('maceworkchain')
RuttleStructureWorkChain = WorkflowFactory('rattleworkchain')
aiida.load_profile()

def get_machine(machine_str):	
	if machine_str == "machine_cm":
		machine = {
		'time'                             : "00:10:00",
		'nodes'                            : 1,
		'mem'                              : "10GB",
		'taskpn'                           : 8,
		'taskps'                           : "1",
		'cpupt'                            : "1",
		'partition'                        : "cm01,cm02,cm03,cm04",
		'gpu'                              : "0",
		'pool'                             : "1",
		'poolx'                            : "1",
		'pools'                            : "1",
		'pooln'                            : "1",
		'poolp'                            : "1",
		}
		
	elif machine_str == "machine_leonardo":
		machine = {
		'time'                             : "00:30:00",
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
	
	if machine['mem'].find('MB') != -1:
		mem = int(machine['mem'].replace('MB','')) * 1024
	elif machine['mem'].find('GB') != -1:
		mem = int(machine['mem'].replace('GB','')) * 1024 * 1024
	elif machine['mem'].find('KB') != -1:
		mem = int(machine['mem'].replace('KB',''))

	time = machine['time'].split(':')
	time_sec = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])

	return machine, time_sec, mem

def set_qe_builder_parameters(builder, builder_str, description, pseudo_family, structures, kpoints, cutoff_wfc, cutoff_rho, time_sec, mem, machine):

	# Setting metadata options
	builder.scf.pw.metadata.options.withmpi = True
	builder.scf.pw.metadata.description = description
	builder.scf.pw.metadata.options.max_wallclock_seconds = time_sec
	builder.scf.pw.metadata.options.import_sys_environment = False
	builder.scf.pw.metadata.options.max_memory_kb = mem

	# Setting pseudos
	builder.scf.pw.pseudos = pseudo_family.get_pseudos(structure=structures[0])

	# Setting kpoints
	builder.scf.kpoints = kpoints

	# Setting pw parameters
	parameters = {
	'SYSTEM': {
	    'ecutwfc': cutoff_wfc,
	    'ecutrho': cutoff_rho,
	    'degauss': 2.2049585400e-02,
	    'occupations': 'smearing',
	    'smearing': 'cold',
	    'nosym': True,
	},
	'CONTROL': {'calculation': 'scf'},
	'ELECTRONS': {
	    'conv_thr': 1.0e-8,
	    'mixing_beta': 0.5,
	    'electron_maxstep': 50,
	    'mixing_mode': 'local-TF',
	}
	}
	builder.scf.pw.parameters = Dict(parameters)

	# Setting resources
	resources = {
	'num_machines': machine["nodes"],
	'num_mpiprocs_per_machine': machine["taskpn"],
	'num_cores_per_mpiproc': machine['cpupt']
	}
	builder.scf.pw.metadata.options.resources = resources

	if builder_str == "machine_leonardo":
		builder.scf.pw.metadata.options.account = machine['account']
		builder.scf.pw.metadata.options.queue_name = machine['partition']
		builder.scf.pw.metadata.options.custom_scheduler_commands=f'#SBATCH --gres=gpu:{machine["gpu"]} '
		builder.scf.pw.metadata.options.qos = machine['qos']

def set_lammps_builder_parameters(builder, description, machine, time_sec, mem):
    # Setting metadata options
    builder.lmp.metadata.options.resources = {
        'num_machines': machine['nodes'],
        'num_mpiprocs_per_machine': machine['taskpn'],
        'num_cores_per_mpiproc': machine['cpupt']
    }
    builder.lmp.metadata.description = description
    builder.lmp.metadata.options.qos = machine.get('qos', None)
    builder.lmp.metadata.options.account = machine.get('account', None)
    builder.lmp.metadata.options.queue_name = machine.get('partition', None)
    builder.lmp.metadata.options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{machine.get('gpu', 0)}"
    builder.lmp.metadata.options.max_wallclock_seconds = time_sec
    builder.lmp.metadata.options.max_memory_kb = mem
    builder.lmp.metadata.options.import_sys_environment = False
    
def set_mace_builder_parameters(builder, description, machine, time_sec, mem):
	# Setting metadata options
	builder.mace.metadata.options.resources = {
	    'num_machines': machine['nodes'],
	    'num_mpiprocs_per_machine': machine['taskpn'],
	    'num_cores_per_mpiproc': machine['cpupt']
	}

	builder.mace.metadata.description = description

	builder.mace.metadata.options.account = machine['account']
	builder.mace.metadata.options.queue_name = machine['partition']
	builder.mace.metadata.options.custom_scheduler_commands=f"#SBATCH --gres=gpu:{machine['gpu']}"
	builder.mace.metadata.options.qos = machine['qos']
	builder.mace.metadata.options.max_wallclock_seconds = time_sec
	builder.mace.metadata.options.max_memory_kb = mem
	builder.mace.metadata.options.import_sys_environment = False


# Set the folder path
folder_path = 'Data/RowStructures'

structure_uuids = load_structures_from_folder(folder_path)
#structures = [load_node(3139), load_node(3153)] #gr 1x1
#structure_uuids = [structure.uuid for structure in structures['structures']]

print("Extracted Structures")
print(structure_uuids)

rattle_params = {
#    'rattle_radius_list'    : [0.1, 0.2, 0.3, 0.4],
#    'sigma_strain_list'     : [0.9, 0.95, 1.00, 1.05, 1.1],
    'rattle_radius_list'    : [0.1],
    'sigma_strain_list'     : [1.00],
    'n_configs'             : 2,
    'frac_vacancies'        : 0.4,
    'vacancies_per_config'  : 1,
    'do_equilibrium'        : True
}

result_rattle_structures = run(RuttleStructureWorkChain, structure_uuids=structure_uuids['uuids'], rattle_params = Dict(rattle_params))

print("result_rattle_structures PK")
print(result_rattle_structures)

mod_structures = []
for structure_entry in result_rattle_structures['structures_parameters_list']:
	structure_uuid = structure_entry['out_structure_pk']
	structure = load_node(structure_uuid)
	mod_structures.append(structure)
	
print("Modified Structures")
print(mod_structures)	


machine_QE, time_sec_QE, mem_QE = get_machine("machine_cm")

description_QE = "test_gr"

kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])

code_QE = load_code('pw@cm01')
#code_QE = load_code('qe7.2-pw@leo2_scratch_bind')
pseudo_family_label = Str('SSSP/1.3/PBE/precision')
pseudo_family = load_group('SSSP/1.3/PBE/precision')
	
cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=mod_structures[0], unit='Ry')

# Set the parameters
builder_QE = QECalculationWorkChain.get_builder_from_protocol(code=code_QE, structure_list=mod_structures)
set_qe_builder_parameters(builder_QE, "machine_cm", description_QE, pseudo_family, mod_structures, kpoints, cutoff_wfc, cutoff_rho, time_sec_QE, mem_QE, machine_QE)

results_QE = run(builder_QE)
print("Results QE")
print(results_QE['dataset_list'])	

# Create or load code
code_mace = load_code("mace@leo2_scratch_bind")
machine_mace, time_sec_mace, mem_mace = get_machine("machine_leonardo")
description_mace = "pot 1.4"

#dataset_mace = load_node(22289) 
dataset_mace = results_QE['dataset_list']
builder_mace = MaceWorkChain.get_builder_from_protocol()
set_mace_builder_parameters(builder_mace, description_mace, machine_mace, time_sec_mace, mem_mace)

#calc_mace = submit(builder_mace,
#                code=code_mace,
#                dataset_list = dataset_mace,
#                parent_folder 	=	Str(Path(__file__).resolve().parent)
#                )
#print("calc_mace['validation list']")
#print(calc_mace)

# Create or load code
code_lammps = load_code("lmp4mace@leo2_scratch_bind")

# Set up inputs
atoms = read(f'Data/RattleStructures/wte2-1t1_9x5.xyz')
structure = StructureData(ase=atoms)
structure.store()
potential=SinglefileData(file=f'/home/nataliia/Documents/aiida_scripts/Data/Potentials/R_1H_1T1_swa.model-lammps.pt')
potential.store()

description_lammps = "relaxation of WTe2 interface 2"
machine_lammps, time_sec_lammps, mem_lammps = get_machine("machine_leonardo")

builder_lammps = LammpsWorkChain.get_builder_from_protocol()
set_lammps_builder_parameters(builder_lammps, description_lammps, machine_lammps, time_sec_lammps, mem_lammps)


#for temp in range(200, 1850, 50):
temp=300
#for structure in mod_structures:
#	calc = submit(builder_lammps,
#		          code			=	code_lammps,
#		          structure		=	structure,
#		          potential		=	potential,
#		          temperature		=	Float(temp),
#		          pressure	    	=	Float(0.0),
#		          dt			=	Float(0.00242),
#		          num_steps		=	Int(200),
#		          parent_folder 	=	Str(Path(__file__).resolve().parent),
#		         )
#	print(f'Running calculation with temp = {temp}; PK = {calc.pk}; structure = {structure.pk}')
