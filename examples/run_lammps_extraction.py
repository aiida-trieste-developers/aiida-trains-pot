from aiida.orm import load_code, load_node, load_group, Str, Dict, Group, Int, Data, Bool, Float
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from aiida.tools.groups import GroupPath

load_profile()

LammpsExtraction = WorkflowFactory('NNIPdevelopement.lammpsextraction')




builder = LammpsExtraction.get_builder()

builder.trajectory = load_node(38534)
builder.input_structure = load_node(36335)
builder.dt = Float(0.00242)
builder.correlation_time = Float(2.42)
builder.saving_frequency = Int(100)

submit(builder) #.id