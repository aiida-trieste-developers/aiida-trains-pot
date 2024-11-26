from aiida.engine import WorkChain, ToContext, append_, calcfunction
from aiida.orm import Float, Dict, List, Int, Bool, FolderData, SinglefileData, StructureData
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.common import AttributeDict
import os
from aiida_lammps.data.potential import LammpsPotentialData
from pathlib import Path
import tempfile


LammpsWorkChain = WorkflowFactory('lammps.base')
PESData         = DataFactory('pesdata')

def generate_potential(potential) -> LammpsPotentialData:
        """
        Generate the potential to be used in the calculation.

        Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

        :return: potential to do the calculation
        :rtype: LammpsPotentialData
        """
        potential_parameters = {
            "species": [],
            "atom_style": "atomic",        
            "units": "metal",
            "extra_tags": {},                
        }

        # Assuming you have a trained MACE model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with potential.open(mode='rb') as potential_handle:
                potential_content = potential_handle.read()
            tmp_file.write(potential_content)
            tmp_file_path = tmp_file.name
            
        potential = LammpsPotentialData.get_or_create(
            #source=binary_stream,
            source = Path(tmp_file_path),
            pair_style="mace no_domain_decomposition",
            **potential_parameters,
        )

        os.remove(tmp_file_path)
    
        return potential
    
class ExplorationWorkChain(WorkChain):
    """A workchain to loop over structures and submit AbInitioLabellingWorkChain."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('params_list', valid_type=List, help='List of parameters for md',)
        spec.input('parameters', valid_type=Dict, help='Global parameters for lammps',)
        spec.input('potential_lammps', valid_type=SinglefileData, help='One of the potential for MD', )
        spec.input_namespace('lammps_input_structures',  valid_type=StructureData, help='Input structures for lammps', )        
        spec.input('sampling_time', valid_type=Float, help='Correlation time for frame extraction', )
         
        spec.expose_inputs(LammpsWorkChain, namespace="md", exclude=('lammps.structure', 'lammps.potential','lammps.parameters'), namespace_options={'validator': None})
        spec.output_namespace("md", dynamic=True, help="Exploration outputs")
          
        spec.outline(            
            cls.run_md,
            cls.finalize_md,          
        )        
                   

    def run_md(self):

        potential = self.inputs.potential_lammps   

        for _, structure in self.inputs.lammps_input_structures.items():
            inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
            inputs.lammps.structure = structure
            inputs.lammps.potential = generate_potential(potential)
            params_list=list(self.inputs.params_list)
            parameters=AttributeDict(self.inputs.parameters)
            parameters.dump.dump_rate = int(self.inputs.sampling_time/parameters.control.timestep)
            for params_md in params_list:            
                parameters.md = dict(params_md)            
                inputs.lammps.parameters = Dict(parameters)                
                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_wc=append_(future))

    def finalize_md(self):        
        md_out = {}        
        for ii, calc in enumerate(self.ctx.md_wc):            
            if calc.exit_status == 0:                
                self.report(f'md_{ii} exit0')                       
                md_out[f'md_{ii}']={el:calc.outputs[el] for el in calc.outputs}           
        
        self.out('md', md_out)
        