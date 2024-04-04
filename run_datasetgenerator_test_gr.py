from aiida.orm import load_code, load_node, load_group, Str, Dict, Group, Int, Data, Bool, Float
from aiida import load_profile
from aiida.engine import submit, run
from aiida.tools.groups import GroupPath
from aiida.plugins import DataFactory

KpointsData = DataFactory("core.array.kpoints")
from DatasetGeneratorWorkChain import DatasetGeneratorWorkChain
load_profile()

machine = {
'time'                             : "00:10:00",
'nodes'                            : 1,
'mem'                              : "10GB",
'taskpn'                           : 8,
'taskps'                           : "1",
'cpupt'                            : "1",
# 'account'                          : "IscrB_DeepVTe2",
'partition'                        : "cm01,cm02,cm03,cm04",
'gpu'                              : "0",
'pool'                             : "1",
'poolx'                            : "1",
'pools'                            : "1",
'pooln'                            : "1",
'poolp'                            : "1",
# 'qos'                              : "normal"
}

description = "test_gr"

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






# structure= [load_node(46114), load_node(46115)] #mote2
structure = [load_node(219)] #gr 1x1
code = load_code('pw@cm01')
# code = load_code('qe-7.2@leonardo_scratch_qe_gpu')
pseudo_family_label = Str('SSSP/1.3/PBE/precision')
pseudo_family = load_group('SSSP/1.3/PBE/precision')


rattle_params = {
    'rattle_radius_list'    : [0.2, 0.1],
    'sigma_strain_list'     : [0.90, 0.95, 1.00, 1.05, 1.10],
    'n_configs'             : 5,
    'frac_vacancies'        : 0.4,
    'vacancies_per_config'  : 1,
    'do_equilibrium'        : True
}



cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure[0], unit='Ry')



builder = DatasetGeneratorWorkChain.get_builder_from_protocol(code=code, structure_list=structure)
# builder.structure = [structure]
# builder.code = code
# builder.pseudo_family_label = pseudo_family_label
builder.rattle_params = Dict(rattle_params)


builder.scf.pw.metadata.options.withmpi=True
builder.scf.pw.metadata.description = description
# builder.scf.pw.pseudos = pseudo_family
builder.scf.pw.pseudos = pseudo_family.get_pseudos(structure=structure[0])
print(builder.scf.pw.pseudos)
builder.scf.kpoints = kpoints
builder.scf.pw.parameters = Dict({'SYSTEM': 
                                  {
                                    'ecutwfc': cutoff_wfc,
                                    'ecutrho': cutoff_rho,
                                    'degauss': 2.2049585400e-02,
                                    'occupations': 'smearing',
                                    'smearing': 'cold',
                                    'nosym': True,
                                   },
                                   'CONTROL': {'calculation': 'scf'},
                                   'ELECTRONS':
                                    {
                                       'conv_thr': 1.0e-8,
                                       'mixing_beta': 0.5,
                                       'electron_maxstep': 50,
                                       'mixing_mode': 'local-TF',
                                    }
                                  })
builder.scf.pw.metadata.options.max_wallclock_seconds = time_sec
builder.scf.pw.metadata.options.import_sys_environment = False
builder.scf.pw.metadata.options.max_memory_kb = mem
builder.scf.pw.metadata.options.resources = {'num_machines': machine["nodes"], 'num_mpiprocs_per_machine': machine["taskpn"], 'num_cores_per_mpiproc': machine['cpupt']}

# if 'leonardo' in code.full_label:
    # builder.scf.pw.metadata.options.account = machine['account']
# builder.scf.pw.metadata.options.queue_name = machine['partition']
# builder.scf.pw.metadata.options.custom_scheduler_commands=f'#SBATCH --gres=gpu:{machine["gpu"]} '
    # builder.scf.pw.metadata.options.qos = machine['qos']

# print(builder.structure_list.numsteps)
#submit(builder) #.id
run(builder)
