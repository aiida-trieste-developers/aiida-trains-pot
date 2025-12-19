"""
AiiDA calculation plugin for the METATrain code.
"""
import re
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, List, FolderData, Dict, Bool, Code, Str
import io
from contextlib import redirect_stdout
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from aiida.plugins import DataFactory
import random
import yaml
import copy

PESData = DataFactory('pesdata')

def validate_protocol(node, _):
    """Validate the protocol input."""
    if node.value not in ["naive-finetune", "replay-finetune"]:
        return "The `protocol` input can only be 'naive-finetune' or 'replay-finetune'."

def validate_inputs(inputs, _):
    """Validate the top-level inputs."""
    if 'protocol' in inputs:
        if inputs["protocol"].value == "naive-finetune" and "finetune_model" not in inputs:
            return "The `finetune_model` input is required when using the 'naive-finetune' protocol."
        if inputs["protocol"].value == "replay-finetune":
            if "finetune_model" not in inputs:
                return "The `finetune_model` input is required when using the 'replay-finetune' protocol."
            if "finetune_replay_dataset" not in inputs:
                return "The `finetune_replay_dataset` input is required when using the 'replay-finetune' protocol."

class MetaTrainCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the diff executable.

    Simple AiiDA plugin wrapper for 'diffing' two files.
    """

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1,}
        spec.inputs["metadata"]["options"]["parser_name"].default = "trains_pot.metatrain"
        #ADD input parameters

        # new ports
        spec.input("training_set", valid_type=PESData, help="Training dataset list",)
        spec.input("validation_set", valid_type=PESData, help="Validation dataset list",)
        spec.input("test_set", valid_type=PESData, help="Test dataset list",)
        spec.input("meta_config", valid_type=Dict, help="Config parameters for METATrain", required=False)
        spec.input("checkpoints", valid_type=FolderData, help="Checkpoints file", required=False)
        spec.input("do_preprocess", valid_type=Bool, help="Perform preprocess", required=False, default=lambda:Bool(False))
        spec.input("preprocess_code", valid_type=Code, help="Preprocess code, required if do_preprocess is True", required=False)
        spec.input("postprocess_code", valid_type=Code, help="Postprocess code", required=False)
        spec.input("restart", valid_type=Bool, help="Restart from a previous calculation", required=False, default=lambda:Bool(False))
        spec.input("checkpoints_restart", valid_type=FolderData, help="Checkpoints file", required=False)
        spec.input("protocol", valid_type=Str, help="Protocol for the calculation {'naive-finetune' or 'replay-finetune'}", required=False, validator=validate_protocol)
        spec.input("finetune_model", valid_type=SinglefileData, help="Model to finetune", required=False)
        spec.input("finetune_replay_dataset", valid_type=PESData, help="Dataset for replay finetune", required=False)
        spec.inputs.validator = validate_inputs     

        #spec.output("model_stage1_lammps", valid_type=SinglefileData, help="Stage 1 model compiled for LAMMPS",)
        #spec.output("model_stage1_ase", valid_type=SinglefileData, help="Stage 1 model compiled for ASE",)
        #spec.output("model_stage1_pytorch", valid_type=SinglefileData, help="Stage 1 model not compiled",)

        spec.output("model_stage2_lammps", valid_type=SinglefileData, help="Stage 2 model compiled for LAMMPS",)
        #spec.output("model_stage2_ase", valid_type=SinglefileData, help="Stage 2 model compiled for ASE",)
        #spec.output("model_stage2_pytorch", valid_type=SinglefileData, help="Stage 2 model not compiled",)
        
        spec.output("meta_out", valid_type=SinglefileData, help="Meta output file",)       
        #spec.output("logs", valid_type=FolderData, help="Logs file",)
        spec.output("checkpoints", valid_type=FolderData, help="Checkpoints file",)
        spec.output("RMSE", valid_type=List, help="List of the checkpoints result table",)
        #spec.output("results", valid_type=List, help="List of the results of log file",)
        

        spec.exit_code(300, "ERROR_MISSING_OUTPUT_FILES", message="Calculation did not produce all expected output files.",)
        spec.exit_code(400, "ERROR_OUT_OF_WALLTIME", message="The calculation stopped prematurely because it ran out of walltime.", )
        
    
    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        meta_config_dict = self.inputs.meta_config.get_dict()
        

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params =f"""train config.yml""".split()    

        

        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "meta.out"
        
        training_txt = self.inputs.training_set.get_txt(write_params=False, key_prefix='dft')
        validation_txt = self.inputs.validation_set.get_txt(write_params=False, key_prefix='dft')
        test_txt = self.inputs.test_set.get_txt(write_params=False, key_prefix='dft')

        with folder.open('training.xyz', "w") as handle:
            handle.write(training_txt)
        with folder.open('validation.xyz', "w") as handle:
            handle.write(validation_txt)
        with folder.open('test.xyz', "w") as handle:
            handle.write(test_txt)

        if 'finetune_model' not in self.inputs:
            if 'training_set' not in meta_config_dict:
                meta_config_dict['training_set'] = {}            
            
            if 'systems' not in meta_config_dict['training_set']:
                meta_config_dict['training_set']['systems'] = {}

            if 'targets' not in meta_config_dict['training_set']:
                meta_config_dict['training_set']['targets'] = {}

            if 'energy' not in meta_config_dict['training_set']['targets']:
                meta_config_dict['training_set']['targets']['energy'] = {}

            meta_config_dict['training_set']['systems']['read_from'] = "training.xyz"
            meta_config_dict['training_set']['systems']['reader'] = "ase"
            meta_config_dict['training_set']['systems']['length_unit'] = "angstrom"

            meta_config_dict['training_set']['targets']['energy']['read_from'] = "training.xyz"
            meta_config_dict['training_set']['targets']['energy']['reader'] = "ase"
            meta_config_dict['training_set']['targets']['energy']['key'] = "dft_energy"
            meta_config_dict['training_set']['targets']['energy']['unit'] = "eV"

            if 'test_set' not in meta_config_dict:
                meta_config_dict['test_set'] = copy.deepcopy(meta_config_dict['training_set'])

            meta_config_dict['test_set']['systems']['read_from'] = "test.xyz"
            meta_config_dict['test_set']['targets']['energy']['read_from'] = "test.xyz"

            if 'validation_set' not in meta_config_dict:
                meta_config_dict['validation_set'] = copy.deepcopy(meta_config_dict['training_set'])

            meta_config_dict['validation_set']['systems']['read_from'] = "validation.xyz"
            meta_config_dict['validation_set']['targets']['energy']['read_from'] = "validation.xyz"
       

        finetune = False
        if 'finetune_model' in self.inputs:
            finetune = True
        

        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(meta_config_dict, yaml_file, default_flow_style=False)
        # Save the checkpoints folder
        if 'checkpoints' in self.inputs:

            codeinfo.cmdline_params =f"""train config.yml --restart checkpoints/model.ckpt""".split()   
            
            checkpoints_folder = self.inputs.checkpoints
            folder.get_subfolder('checkpoints', create=True)  # Create the checkpoints directory
            for checkpoint_file in checkpoints_folder.list_object_names():
                with checkpoints_folder.open(checkpoint_file, 'rb') as source:                    
                    with folder.open(f'checkpoints/model.ckpt', 'wb') as destination:
                        destination.write(source.read())

        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(meta_config_dict, yaml_file, default_flow_style=False)
        calcinfo = datastructures.CalcInfo()
        if finetune:            
            calcinfo.local_copy_list = [
                (
                    self.inputs.finetune_model.uuid,
                    self.inputs.finetune_model.filename,
                    "finetune_model.ckpt",
                ),]
        
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['model.ckpt', 'meta.out', 'model.pt', '_scheduler-std*', 'outputs']

        return calcinfo
