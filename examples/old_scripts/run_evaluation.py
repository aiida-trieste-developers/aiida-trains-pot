# -*- coding: utf-8 -*-
"""Launch a calculation using the 'diff-tutorial' plugin"""
from pathlib import Path
import aiida
from aiida import engine, orm
from aiida.common.exceptions import NotExistent
from aiida.engine import submit
from aiida.plugins import CalculationFactory
from aiida.orm import load_node

aiida.load_profile()

evaluation = CalculationFactory('NNIPdevelopment.evaluation')




machine = {
'time'                             : "01:00:00",
'nodes'                            : 1,
'mem'                              : "7GB",
'taskpn'                           : 8,
'taskps'                           : "1",
'cpupt'                            : "1",
'account'                          : "IscrB_DeepVTe2",
'partition'                        : "main",
'gpu'                              : "1",
'pool'                             : "1",
'poolx'                            : "1",
'pools'                            : "1",
'pooln'                            : "1",
'poolp'                            : "1",
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

from pathlib import Path
from aiida.orm.nodes.data.code.portable import PortableCode
# code = PortableCode(
#     label='test-evaluation',
#     default_calc_job_plugin = "test-evaluation",
#     description='A description for the code',
#     filepath_files=Path('/home/bidoggia/onedrive/aiida/git/NNIPDevelopment/src/NNIPdevelopment/evaluation/'),
#     filepath_executable='cometee_evaluation.py'
# )
# code._get_cli_options
# print(code.get_executable())
# print(code._get_cli_options)

code = orm.load_code("cometee-evaluation@bora")



builder = code.get_builder()

# builder.metadata.computer=orm.load_computer("bora")

builder.metadata.options.resources = {
    'num_machines': machine['nodes'],
    'num_mpiprocs_per_machine': machine['taskpn'],
    'num_cores_per_mpiproc': machine['cpupt']
}

builder.metadata.description = "pot 1.4"
builder.metadata.options.qos = machine['qos']
builder.metadata.options.queue_name = machine['partition']
builder.metadata.options.custom_scheduler_commands=f"#SBATCH --gres=gpu:{machine['gpu']}"
builder.metadata.options.max_wallclock_seconds = time_sec
builder.metadata.options.max_memory_kb = mem
builder.metadata.options.import_sys_environment = False
builder.metadata.description = 'Mace error calculation'


submit(builder, mace_potentials={'pot_1': load_node(51292), 'pot_2': load_node(51295), 'pot_3': load_node(51280), 'pot_4':load_node(51302)}, datasetlist=load_node(51334))