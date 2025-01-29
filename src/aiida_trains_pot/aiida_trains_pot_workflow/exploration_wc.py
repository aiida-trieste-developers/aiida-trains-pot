from aiida.engine import WorkChain, ToContext, append_, calcfunction, while_
from aiida.orm import Float, Dict, List, Str, SinglefileData, StructureData
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.common import AttributeDict
import os
from aiida_lammps.data.potential import LammpsPotentialData
from pathlib import Path
import tempfile
import random  # to change seed for each retry

LammpsWorkChain = WorkflowFactory('lammps.base')
PESData         = DataFactory('pesdata')

def generate_potential(potential, pair_style) -> LammpsPotentialData:
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
        source = Path(tmp_file_path),
        pair_style = pair_style,
        **potential_parameters,
    )

    os.remove(tmp_file_path)
    
    return potential


class ExplorationWorkChain(WorkChain):
    """A workchain to loop over structures and submit LammpsWorkChain with retries."""


    ###################################################################
    ##                       DEFAULT VALUES                          ##
    ###################################################################

    DEFAULT_potential_pair_style = Str('mace no_domain_decomposition')
    ###################################################################

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('params_list', valid_type=List, help='List of parameters for md')
        spec.input('parameters', valid_type=Dict, help='Global parameters for lammps')
        spec.input('potential_lammps', valid_type=SinglefileData, required=False, help='One of the potential for MD')
        spec.input('potential_pair_style', valid_type=Str, default=lambda:cls.DEFAULT_potential_pair_style, required=False, help=f"General potential pair style. Default: {cls.DEFAULT_potential_pair_style}")
        spec.input_namespace('lammps_input_structures', valid_type=StructureData, help='Input structures for lammps')
        spec.input('sampling_time', valid_type=Float, help='Correlation time for frame extraction')

        spec.expose_inputs(LammpsWorkChain, namespace="md", exclude=('lammps.structure', 'lammps.potential', 'lammps.parameters'), namespace_options={'validator': None})
        spec.output_namespace("md", dynamic=True, help="Exploration outputs")
        
        spec.outline(            
            cls.run_md,
            cls.finalize_md,          
        )  
        spec.outline(            
            cls.run_md,
            while_(cls.not_converged)(
                cls.run_restart,
            ),
            cls.finalize_md,            
        )     

    def run_md(self):
        """Run MD simulations for each structure and MD parameter set, with retries on failure."""
        potential = self.inputs.potential_lammps    
        self.ctx.rerun_wc = [] 
        self.ctx.rerun_wc_old = [] 
        self.ctx.last_wc = []
        self.ctx.dict_wc = {} 
        self.ctx.iteration = 0     
        # Loop over structures
        for _, structure in self.inputs.lammps_input_structures.items():
            inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
            inputs.lammps.structure = structure
            inputs.lammps.potential = generate_potential(potential, str(self.inputs.potential_pair_style.value))
            
            params_list = list(self.inputs.params_list)
            parameters = AttributeDict(self.inputs.parameters)
            parameters.dump.dump_rate = int(self.inputs.sampling_time / parameters.control.timestep)
            
            # Loop over the MD parameter sets
            for params_md in params_list:
                
                parameters.md = dict(params_md)
                inputs.lammps.parameters = Dict(parameters)
                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_wc=append_(future))                 
                self.ctx.dict_wc[f'{self.ctx.iteration}'] = self.ctx.iteration 
                self.ctx.last_wc.append(self.ctx.iteration)
                self.ctx.iteration += 1                     
               

    def run_restart(self):
        self.ctx.last_wc= []
        for ii, calc in enumerate(self.ctx.md_wc):
            if (ii in self.ctx.rerun_wc):                
                incoming = calc.base.links.get_incoming().nested()
            
                # Build the inputs dictionary
                inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
                for key, node in incoming.items():
                    if key == 'lammps':
                        inputs[key].update(node)  # Merge nested inputs
                    
                
                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_wc=append_(future))
                self.ctx.dict_wc[f'{self.ctx.iteration}'] = self.ctx.dict_wc[f'{ii}']
                self.ctx.last_wc.append(self.ctx.iteration)
                self.ctx.iteration += 1          
        
    def not_converged(self):
        """Check if any calculation did not end successfully and requires a restart."""
        # Update the old list of reruns and prepare a new one
        self.ctx.rerun_wc_old.extend(self.ctx.rerun_wc)
        self.ctx.rerun_wc = []
    

        for ii, calc in enumerate(self.ctx.md_wc):
            if (ii in self.ctx.last_wc) and calc.exit_status != 0:
                # Count how many times the current calculation has been retried
                retry_count = sum(1 for value in self.ctx.dict_wc.values() if value == self.ctx.dict_wc[f'{ii}'])                
                
                # Check if the calculation failed and has been retried less than 5 times
                if retry_count < 3 and (ii not in self.ctx.rerun_wc_old):
                    self.ctx.rerun_wc.append(ii)

        return len(self.ctx.rerun_wc) > 0      

    def finalize_md(self):
        """Collect the results from the completed LAMMPS calculations."""
        md_out = {}
        for ii, calc in enumerate(self.ctx.md_wc):
            if calc.exit_status == 0:
                self.report(f'md_{ii} exit0')
                md_out[f'md_{ii}'] = {el: calc.outputs[el] for el in calc.outputs}
        
        self.out('md', md_out)
