from aiida.engine import WorkChain, ToContext, append_, calcfunction
from aiida.orm import StructureData, Dict, Str, Group, load_group
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.common import AttributeDict
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PESData         = DataFactory('pesdata')

@calcfunction
def WriteLabelledDataset(non_labelled_structures, **labelled_data):
    labelled_dataset = []
    elem_charge = 1.60217653e-19
    gpa_to_eV_per_ang3 = 1.0e9/elem_charge/1.0e30
    for key, value in labelled_data.items():
        labelled_dataset.append(non_labelled_structures.get_list()[int(key.split('_')[1])])
        labelled_dataset[-1]['dft_energy'] = float(value['output_parameters'].dict.energy)
        labelled_dataset[-1]['dft_forces'] = value['output_trajectory'].get_array('forces')[0].tolist()
        labelled_dataset[-1]['dft_stress'] = value['output_trajectory'].get_array('stress')[0]*gpa_to_eV_per_ang3.tolist()

    pes_labelled_dataset = PESData(labelled_dataset)        
    return pes_labelled_dataset


class AbInitioLabellingWorkChain(WorkChain):
    """A workchain to loop over structures and submit AbInitioLabellingWorkChain."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('unlabelled_dataset', valid_type=PESData, help="Structures to label.")     
        spec.input('group_label', valid_type=Str, help="Label for group.", required=False)        
        spec.expose_inputs(PwBaseWorkChain, namespace="quantumespresso", exclude=('pw.structure',), namespace_options={'validator': None})          
        spec.output("ab_initio_labelling_data", valid_type=PESData,)
        spec.outline(
            cls.setup,
            cls.run_ab_initio_labelling,
            cls.finalize            
        )        
        

    def setup(self):
        """Initialize context and input parameters."""                
        # Initialize the list of structures
        self.ctx.config = 0        

    def run_ab_initio_labelling(self):
        """Run PwBaseWorkChain for each structure."""

        # Create or load a group to track the calculations
        if hasattr(self.inputs, 'group_label'):
            group_label = self.inputs.group_label.value
            
        else:
            group_label = f'ab_initio_labelling_{self.uuid}'
            self.report(f'Saving configurations in group {group_label}')

        try:
            group = load_group(group_label)
            self.report(f'Using existing group: {group_label}')
        except:
            group = Group(label=group_label).store()
            self.report(f'Created new group: {group_label}')
        
        for _, structure in enumerate(self.inputs.unlabelled_dataset.get_ase_list()):   
            self.ctx.config += 1    
            str_data = StructureData(ase=structure)

            # Prepare inputs
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='quantumespresso'))
            inputs.pw.structure = str_data
            inputs.metadata.call_link_label = f'ab_initio_labelling_config_{self.ctx.config}'
            

            atm_types = list(str_data.get_symbols_set())
            pseudos = inputs.pw.pseudos
            inputs.pw.pseudos = {}
            for tp in atm_types:
                if tp in pseudos.keys():
                    inputs.pw.pseudos[tp] = pseudos[tp]
                else:
                    raise ValueError(f'Pseudopotential for {tp} not found')
                
                        
            default_inputs = {'CONTROL': {'calculation': 'scf', 'tstress': True, 'tprnfor': True}}
            inputs.pw.parameters = Dict(recursive_merge(default_inputs, inputs.pw.parameters.get_dict()))
            
            inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
            # Submit the workchain
            future = self.submit(PwBaseWorkChain, **inputs)
            self.report(f'Launched AbInitioLabellingWorkChain for configuration {self.ctx.config} <{future.pk}>')

            # Add the calculation to the group
            group.add_nodes(future)
            self.to_context(ab_initio_labelling_calculations=append_(future))            

    def finalize(self):

        ab_initio_labelling_data = {}
        for ii, calc in enumerate(self.ctx.ab_initio_labelling_calculations):
            if calc.exit_status == 0:
                ab_initio_labelling_data[f'abinitiolabelling_{ii}'] = {
                    'output_parameters': calc.outputs.output_parameters,
                    'output_trajectory': calc.outputs.output_trajectory
                    }
                
        pes_dataset_out = WriteLabelledDataset(non_labelled_structures = self.inputs.unlabelled_dataset, **ab_initio_labelling_data)
                
                
        self.out("ab_initio_labelling_data", pes_dataset_out)
