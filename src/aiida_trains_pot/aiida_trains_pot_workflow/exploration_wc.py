from aiida.engine import WorkChain, ToContext, append_, calcfunction, while_
from aiida.orm import Float, Dict, List, Str, SinglefileData, StructureData
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.common import AttributeDict
import os
from aiida_lammps.data.potential import LammpsPotentialData
from pathlib import Path
import tempfile
import random  # to change seed for each retry
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_trains_pot.utils.lammps_pair_coeffs import get_dftd2_pair_coeffs, get_mace_pair_coeff, get_meta_pair_coeff

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

###################################################################
##                       DEFAULT VALUES                          ##
###################################################################

DEFAULT_potential_pair_style = Str("mace no_domain_decomposition")
DEFAULT_settings             = Dict({"store_restart": True,},) 
DEFAULT_parameters           = Dict({"structure":{
                                        "atom_style" : "atomic",
                                        "atom_modify": "map yes",
                                        "dimension"  : "3",
                                        "boundary"   : "p p p",
                                        },
                                    "potential":{
                                        "potential_style_options": "mace no_domain_decomposition",
                                        },
                                    "control":{
                                        "timestep": 0.001,
                                        "newton": 'on',
                                        "units": 'metal',
                                        },
                                    "thermo":{
                                        "printing_rate": 100,
                                        "thermo_printing": {
                                            "step": True,
                                            "time": True,
                                            "pe": True,
                                            "ke": True,
                                            "etotal": True,
                                            "press": True,
                                            "pxx": True,
                                            "pyy": True,
                                            "pzz": True,
                                            "temp": True,
                                            },
                                        },
                                    "restart":
                                        {"print_final": True,
                                        },
                                    "md": {},
                                    "dump": {},
                                    })
###################################################################

class ExplorationWorkChain(WorkChain):
    """A workchain to loop over structures and submit LammpsWorkChain with retries."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('params_list', valid_type=List, help='List of parameters for md')
        spec.input('parameters', valid_type=Dict, help='Global parameters for lammps')
        spec.input('potential_lammps', valid_type=SinglefileData, required=False, help='One of the potential for MD')
        spec.input('potential_pair_style', valid_type=Str, default=lambda:DEFAULT_potential_pair_style, required=False, help=f"General potential pair style. Default: {DEFAULT_potential_pair_style}")
        spec.input('sampling_time', valid_type=Float, help='Correlation time for frame extraction')
        spec.input('protocol', valid_type=Str, help='Protocol for the calculation', required=False)
        spec.input('lammps_input_structures', valid_type=PESData, help='Input structures for lammps')
        

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
        for structure in self.inputs.lammps_input_structures.get_ase_list():
            inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
            inputs.lammps.structure = StructureData(ase=structure)
            inputs.lammps.potential = generate_potential(potential, str(self.inputs.potential_pair_style.value))

            generate_pair_coeff = True
            if "potential" in self.inputs.parameters:
                if "pair_coeff_list" in self.inputs.parameters["potential"]:
                    generate_pair_coeff = False
            
            # Pair coefficients for MACE potential without hybrid/overlay is always generated, if needed it is overwritten
            if generate_pair_coeff:               
                if 'metatomic' in self.inputs.potential_pair_style.value:
                    pair_coeffs = [get_meta_pair_coeff(inputs.lammps.structure, hybrid=False)]
                else:
                    pair_coeffs = [get_mace_pair_coeff(inputs.lammps.structure, hybrid=False)]
                
            params_list = self.inputs.params_list.get_list()
            input_parameters = self.inputs.parameters.get_dict()
            if 'metatomic' in self.inputs.potential_pair_style.value:
                if 'potential' in self.inputs.parameters:
                    if 'potential_style_options' not in self.inputs.parameters['potential']:
                        input_parameters['potential']['potential_style_options'] = str('potential.dat')              
            if self.inputs.protocol is not None:
                if self.inputs.protocol == 'vdw_d2':
                    if 'potential' in self.inputs.parameters:
                        if 'potential_style_options' not in self.inputs.parameters['potential']:
                            input_parameters['potential']['potential_style_options'] = 'mace no_domain_decomposition momb 20.0 0.75 20.0'
                    else:
                        input_parameters['potential'] = {'potential_style_options': 'mace no_domain_decomposition momb 20.0 0.75 20.0'}
                    if generate_pair_coeff:
                        # Generate DFT-D2 pair coefficients, it overwrites the MACE pair_coeff generated above
                        pair_coeffs = get_dftd2_pair_coeffs(inputs.lammps.structure)                        
                        pair_coeffs.append(get_mace_pair_coeff(inputs.lammps.structure, hybrid=True))
                        #if 'metatomic' in self.inputs.potential_pair_style.value:
                        #    pair_coeffs.append(get_meta_pair_coeff(inputs.lammps.structure, hybrid=True))                        
            input_parameters['potential']['pair_coeff_list'] = pair_coeffs

   
            parameters = recursive_merge(DEFAULT_parameters.get_dict(), input_parameters)
            lammps_inputs = self.inputs.md.lammps
            if "settings" in lammps_inputs:                
                inputs.lammps.settings = recursive_merge(DEFAULT_settings.get_dict(), self.inputs.md.lammps.settings.get_dict())
            else:
                inputs.lammps.settings = DEFAULT_settings.get_dict()
            # if 'dump' not in parameters:
            #     parameters['dump'] = {}
            parameters['dump']['dump_rate'] = int(self.inputs.sampling_time / parameters['control']['timestep'])
            
            
            # Loop over the MD parameter sets
            for params_md in params_list:           
                
                if not any(inputs.lammps.structure.pbc):
                    params_md["integration"]["style"] = "nvt"

                constraint = params_md["integration"]["constraints"]

                # Map dimensions to constraints
                axes = ["x", "y", "z"]

                # Remove constraints for non-periodic directions
                for idx, axis in enumerate(axes):
                    if not inputs.lammps.structure.pbc[idx]:
                        constraint.pop(axis, None)  # Avoid KeyError if axis doesn't exist

                params_md["integration"]["constraints"] = constraint

                parameters['md'] = dict(params_md)

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
                md_out[f'md_{ii}'] = {el: calc.outputs[el] for el in calc.outputs}
        
        self.out('md', md_out)
