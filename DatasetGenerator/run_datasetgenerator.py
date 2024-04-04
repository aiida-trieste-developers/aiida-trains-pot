from aiida.orm import load_code, load_node, load_group, Str, Dict, Group, Int, Data, Bool, Float
from aiida import load_profile
from aiida.engine import submit
from aiida.tools.groups import GroupPath
from aiida.plugins import DataFactory

KpointsData = DataFactory("core.array.kpoints")
from DatasetGeneratorWorkChaintest2 import DatasetGeneratorWorkChain
load_profile()

machine = {
'time'                             : "01:00:00",
'nodes'                            : 1,
'mem'                              : "7600MB",
'taskpn'                           : 8,
'taskps'                           : "1",
'cpupt'                            : "1",
'account'                          : "IscrB_DeepVTe2",
'partition'                        : "lrd_all_serial",
'gpu'                              : "0",
'pool'                             : "1",
'poolx'                            : "1",
'pools'                            : "1",
'pooln'                            : "1",
'poolp'                            : "1",
'qos'                              : "normal"
}

description = "test_gr3"

if machine['mem'].find('MB') != -1:
    mem = int(machine['mem'].replace('MB',''))*1024
elif machine['mem'].find('GB') != -1:
    mem = int(machine['mem'].replace('GB',''))*1024*1024
elif machine['mem'].find('KB') != -1:
    mem = int(machine['mem'].replace('KB',''))
time = machine['time'].split(':')
time_sec=int(time[0])*3600+int(time[1])*60+int(time[2])


kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])






structure = load_node(14232) #gr 8x8
# structure = load_node(3213) #gr 1x1
# code = load_code('qe-7.2-pw-serial@leonardo6')
code = load_code('qe@cm01')
pseudo_family_label = Str('SSSP/1.2/PBE/efficiency')
pseudo_family = load_group('SSSP/1.2/PBE/efficiency')


rattle_params = {
    'rattle_radius_list'    : [0.02, 0.03],
    'sigma_strain_list'     : [0.98, 1.0, 1.02],
    'n_configs'             : 10,
    'frac_vacancies'        : 0.2,
    'vacancies_per_config'  : 2,
    'do_equilibrium'        : True
}


cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure, unit='Ry')



builder = DatasetGeneratorWorkChain.get_builder_from_protocol(code=code, structure_list=[structure])
# builder.structure = [structure]
# builder.code = code
# builder.pseudo_family_label = pseudo_family_label
builder.rattle_params = Dict(rattle_params)


builder.scf.pw.metadata.options.withmpi=True
builder.scf.pw.metadata.description = description
# builder.scf.pw.pseudos = pseudo_family
builder.scf.pseudos = pseudo_family.get_pseudos(structure=structure)
builder.scf.kpoints = kpoints
builder.scf.pw.parameters = Dict({'SYSTEM': 
                                  {
                                    'ecutwfc': cutoff_wfc,
                                    'ecutrho': cutoff_rho,
                                    'degauss': 7.3498e-03,
                                    'occupations': 'smearing',
                                    'smearing': 'cold',
                                   },
                                   'CONTROL': {'calculation': 'scf'}})
builder.scf.pw.metadata.options.max_wallclock_seconds = time_sec
builder.scf.pw.metadata.options.import_sys_environment = False
builder.scf.pw.metadata.options.max_memory_kb = mem
builder.scf.pw.metadata.options.resources = {'num_machines': machine["nodes"], 'num_mpiprocs_per_machine': machine["taskpn"], 'num_cores_per_mpiproc': machine['cpupt']}

if 'leonardo' in code.full_label:
    builder.scf.pw.metadata.options.account = machine['account']
    builder.scf.pw.metadata.options.queue_name = machine['partition']
    builder.scf.pw.metadata.options.custom_scheduler_commands=f'#SBATCH --gres=gpu:{machine["gpu"]} '
    builder.scf.pw.metadata.options.qos = machine['qos']

print(builder.structure_list.numsteps)
submit(builder) #.id