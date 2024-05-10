# -*- coding: utf-8 -*-
"""Launch a calculation using the 'diff-tutorial' plugin"""
from pathlib import Path
import aiida
from aiida.plugins import WorkflowFactory
from ase.io import read
from aiida.engine import submit, run
from aiida.orm import load_group, Float, load_code, Int, Str, load_node
import numpy as np

#MaceWorkChain = WorkflowFactory('maceworkchain')
from MaceWorkChain.MaceWorkChain import MaceWorkChain

aiida.load_profile()

# Create or load code
code = load_code("mace@leo2_scratch_bind")
# try:
#     code = orm.load_code('diff@localhost')
# except NotExistent:
#     # Setting up code via python API (or use "verdi code setup")
#     code = orm.Code(label='diff', remote_computer_exec=[computer, '/usr/bin/diff'], input_plugin_name='diff-tutorial')

# Set up inputs

# atoms = read('/home/bidoggia/onedrive/ML/WTe2/ref_structures/interface-x2.xyz')
# structure = StructureData(ase=atoms)
# structure.store()
# macecalc = load_node(71415)
# retrived = macecalc.get_retrieved_node()
# with retrived.open('aiida_swa.model-lammps.pt', 'rb') as handle:
# 	potential = SinglefileData(file=handle)
# # potential=SinglefileData(file=f'/home/bidoggia/onedrive/ML/WTe2/MACE/RET-R_1H_1T1+R_inter+LMP-MACE_1H_1T1/potential.dat')
# potential.store()


machine = {
'time'                             : "00:30:00",
'nodes'                            : 1,
'mem'                              : "70GB",
'taskpn'                           : 1,
'taskps'                           : "1",
'cpupt'                            : "8",
'account'                          : "IscrB_DeepVTe2",
'partition'                        : "boost_usr_prod",
#'partition'                        : "lrd_all_serial",
'gpu'                              : "1",
'pool'                             : "1",
'poolx'                            : "1",
'pools'                            : "1",
'pooln'                            : "1",
'poolp'                            : "1",
#'qos'                              : "normal"
'qos'                              : "boost_qos_dbg"
}


if machine['mem'].find('MB') != -1:
    mem = int(machine['mem'].replace('MB',''))*1024
elif machine['mem'].find('GB') != -1:
    mem = int(machine['mem'].replace('GB',''))*1024*1024
elif machine['mem'].find('KB') != -1:
    mem = int(machine['mem'].replace('KB',''))
time = machine['time'].split(':')
time_sec=int(time[0])*3600+int(time[1])*60+int(time[2])



# structure_group = load_group(label = 'wte2/str/1t1-9x5')
# structure_group = load_group(label = 'wte2/str/1h-10x10')
# structure_group = load_group(label = 'wte2/str/interface3-2x2')
# structure_group = load_group(label = 'WTe2/1H/20x20/vacTe1')
# nnip_group = load_group(label = 'WTe2/nnip')


# if len(structure_group.nodes) == 0:
#     raise ValueError('No structures found in group')
# elif len(structure_group.nodes) > 1:
#     raise ValueError('More than one structure found in group')

# if len(nnip_group.nodes) == 0:
#     raise ValueError('No potentials found in group')
# elif len(nnip_group.nodes) > 1:
#     raise ValueError('More than one potential found in group')

# potential = nnip_group.nodes[0]
# structure = structure_group.nodes[0]

dataset = load_node(22888)
#dataset = load_node(22289) 
#print("dataset")
#print(dataset)

builder = MaceWorkChain.get_builder_from_protocol()

builder.mace.metadata.options.resources = {
    'num_machines': machine['nodes'],
    'num_mpiprocs_per_machine': machine['taskpn'],
    'num_cores_per_mpiproc': machine['cpupt']
}

builder.mace.metadata.description = "pot 1.4"
# builder.mace.metadata.options.parser_name = "lammps_base"
builder.mace.metadata.options.qos = machine['qos']

builder.mace.metadata.options.account = machine['account']
builder.mace.metadata.options.queue_name = machine['partition']
builder.mace.metadata.options.custom_scheduler_commands=f"#SBATCH --gres=gpu:{machine['gpu']}"
builder.mace.metadata.options.qos = machine['qos']
builder.mace.metadata.options.max_wallclock_seconds = time_sec
builder.mace.metadata.options.max_memory_kb = mem
builder.mace.metadata.options.import_sys_environment = False


print(builder)

calc = submit(builder,
                code		=	code,
                dataset_list 	= 	dataset,
                parent_folder 	=	Str(Path(__file__).resolve().parent)
                )

# run(builder, code=code, structure=structure, potential=potential, temperature=Float(200.0), dt=Float(0.00242))
