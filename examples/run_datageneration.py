from aiida.orm import load_code, load_node, load_group, Str, Dict, Group, Int, Data, Bool, Float
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from aiida.tools.groups import GroupPath

DatasetGeneratorWorkChain = WorkflowFactory('NNIPdevelopment.datageneration')
load_profile()

structures = [load_node(25538), load_node(25536)] #wte2

builder = DatasetGeneratorWorkChain.get_builder_with_structures(structures)

builder.rattle.params.rattle_fraction = Float(0.1)
builder.rattle.params.max_sigma_strain = Float(0.1)
builder.rattle.params.n_configs = Int(10)
builder.rattle.params.frac_vacancies = Float(0.1)
builder.rattle.params.vacancies_per_config = Int(1)
builder.do_rattle = Bool(True)
builder.do_input = Bool(True)

submit(builder) #.id