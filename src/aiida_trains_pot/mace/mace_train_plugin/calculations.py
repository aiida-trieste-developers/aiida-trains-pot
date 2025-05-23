"""
AiiDA calculation plugin for the MACE training code.
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

class MaceTrainCalculation(CalcJob):
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
        spec.inputs["metadata"]["options"]["parser_name"].default = "trains_pot.macetrain"
        #ADD input parameters

        # new ports
        spec.input("training_set", valid_type=PESData, help="Training dataset list",)
        spec.input("validation_set", valid_type=PESData, help="Validation dataset list",)
        spec.input("test_set", valid_type=PESData, help="Test dataset list",)
        spec.input("mace_config", valid_type=Dict, help="Config parameters for MACE",)
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

        spec.output("model_stage1_lammps", valid_type=SinglefileData, help="Stage 1 model compiled for LAMMPS",)
        spec.output("model_stage1_ase", valid_type=SinglefileData, help="Stage 1 model compiled for ASE",)
        spec.output("model_stage1_pytorch", valid_type=SinglefileData, help="Stage 1 model not compiled",)

        spec.output("model_stage2_lammps", valid_type=SinglefileData, help="Stage 2 model compiled for LAMMPS",)
        spec.output("model_stage2_ase", valid_type=SinglefileData, help="Stage 2 model compiled for ASE",)
        spec.output("model_stage2_pytorch", valid_type=SinglefileData, help="Stage 2 model not compiled",)
        
        spec.output("mace_out", valid_type=SinglefileData, help="Mace output file",)       
        spec.output("logs", valid_type=FolderData, help="Logs file",)
        spec.output("checkpoints", valid_type=FolderData, help="Checkpoints file",)
        spec.output("RMSE", valid_type=List, help="List of the checkpoints result table",)
        spec.output("results", valid_type=List, help="List of the results of log file",)
        

        spec.exit_code(300, "ERROR_MISSING_OUTPUT_FILES", message="Calculation did not produce all expected output files.",)
        spec.exit_code(400, "ERROR_OUT_OF_WALLTIME", message="The calculation stopped prematurely because it ran out of walltime.", )
        
    
    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        mace_config_dict = self.inputs.mace_config.get_dict()
        do_preprocess = self.inputs.do_preprocess.value
        if do_preprocess:
            if 'preprocess_code' not in self.inputs:
                raise ValueError("Preprocess code is required if do_preprocess is True")
            preprocess_code = self.inputs.preprocess_code
        

        
        if do_preprocess:
            codeinfo_preprocess = datastructures.CodeInfo()
            codeinfo_preprocess.code_uuid = preprocess_code.uuid
            codeinfo_preprocess.cmdline_params = [
                '--train_file', "training.xyz",
                '--valid_file', "validation.xyz",
                '--test_file', "test.xyz",
                '--energy_key', "dft_energy",
                '--forces_key', "dft_forces",
                '--stress_key', "dft_stress",
                '--compute_statistics',
                '--h5_prefix', "processed_data/",
                '--seed', str(random.randint(0, 10000))
            ]
            if 'r_max' in mace_config_dict:
                codeinfo_preprocess.cmdline_params += ['--r_max', str(mace_config_dict['r_max'])]

        # for MACE < 0.3.7
        codeinfo_postprocess1 = datastructures.CodeInfo()
        codeinfo_postprocess1.code_uuid = self.inputs.postprocess_code.uuid
        codeinfo_postprocess1.cmdline_params = ["aiida_swa.model"]
        # for MACE >= 0.3.7
        codeinfo_postprocess1b = datastructures.CodeInfo()
        codeinfo_postprocess1b.code_uuid = self.inputs.postprocess_code.uuid
        codeinfo_postprocess1b.cmdline_params = ["aiida_stagetwo.model"]
        
        codeinfo_postprocess2 = datastructures.CodeInfo()
        codeinfo_postprocess2.code_uuid = self.inputs.postprocess_code.uuid
        codeinfo_postprocess2.cmdline_params = ["aiida.model"]

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params =f"""--config config.yml""".split()                 

        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "mace.out"
        
        training_txt = self.inputs.training_set.get_txt(write_params=False, key_prefix='dft')
        validation_txt = self.inputs.validation_set.get_txt(write_params=False, key_prefix='dft')
        test_txt = self.inputs.test_set.get_txt(write_params=False, key_prefix='dft')

        with folder.open('training.xyz', "w") as handle:
            handle.write(training_txt)
        with folder.open('validation.xyz', "w") as handle:
            handle.write(validation_txt)
        with folder.open('test.xyz', "w") as handle:
            handle.write(test_txt)

        
        
        mace_config_dict['seed'] = random.randint(0, 10000)
        if do_preprocess:
            mace_config_dict['train_file'] = "processed_data/train/"   
            mace_config_dict['valid_file'] = "processed_data/val/"
            mace_config_dict['test_file'] = "processed_data/test/"    
            mace_config_dict['statistics_file'] = "processed_data/statistics.json"
        else:
            mace_config_dict['train_file'] = "training.xyz"   
            mace_config_dict['valid_file'] = "validation.xyz"
            mace_config_dict['test_file'] = "test.xyz"

        mace_config_dict['energy_key'] = "dft_energy" 
        mace_config_dict['forces_key'] = "dft_forces" 
        mace_config_dict['stress_key'] = "dft_stress"  

        if 'protocol' in self.inputs:
            if self.inputs.protocol.value == "naive-finetune":
                mace_config_dict['foundation_model'] = "finetune_model.dat"
                mace_config_dict['multiheads_finetuning'] = False
            if self.inputs.protocol.value == "replay-finetune":
                mace_config_dict['foundation_model'] = "finetune_model.dat"
                mace_config_dict['multiheads_finetuning'] = True
                replay_txt = self.inputs.finetune_replay_dataset.get_txt(write_params=False, key_prefix='dft')
                with folder.open('replay.xyz', "w") as handle:
                    handle.write(replay_txt)
                mace_config_dict['pt_train_file'] = "replay.xyz"
                if 'E0s' in mace_config_dict and mace_config_dict['E0s'] == "average":
                    del mace_config_dict['E0s']


        if 'checkpoints' in self.inputs:
            mace_config_dict['restart_latest'] = True

        # for training_structure in self.inputs.training_set:
        #     training_dict = dict(training_structure)
        #     if len(training_dict['symbols']) != 1:
        #         mace_config_dict['E0s'] = "average"
        #         break

        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(mace_config_dict, yaml_file, default_flow_style=False)

        if not mace_config_dict.get('distributed', False) and self.inputs["metadata"]["options"]["resources"].get('num_mpiprocs_per_machine') > 1:
            mace_config_dict['distributed'] = True

        # Save the checkpoints folder
        if 'checkpoints' in self.inputs and self.inputs.restart.value==True:
            mace_config_dict['restart_latest'] = True
            checkpoints_folder = self.inputs.checkpoints
            folder.get_subfolder('checkpoints', create=True)  # Create the checkpoints directory
            for checkpoint_file in checkpoints_folder.list_object_names():

                if '_epoch' in checkpoint_file and '_swa':
                    with checkpoints_folder.open(checkpoint_file, 'rb') as source:
                        new_checkpoint_file = f"aiida_run-{str(mace_config_dict['seed'])}_epoch-0_swa.pt"
                        with folder.open(f'checkpoints/{new_checkpoint_file}', 'wb') as destination:
                            destination.write(source.read())
                elif '_epoch' in checkpoint_file:
                    with checkpoints_folder.open(checkpoint_file, 'rb') as source:
                        new_checkpoint_file = f"aiida_run-{str(mace_config_dict['seed'])}_epoch-0.pt"
                        with folder.open(f'checkpoints/{new_checkpoint_file}', 'wb') as destination:
                            destination.write(source.read())

        if 'checkpoints_restart' in self.inputs:
            mace_config_dict['restart_latest'] = True
            checkpoints_folder = self.inputs.checkpoints_restart
            folder.get_subfolder('checkpoints', create=True)  # Create the checkpoints directory
            for checkpoint_file in checkpoints_folder.list_object_names():
                if '_epoch' in checkpoint_file and '_swa':
                    # Regular expression to extract the seed (assumed to be numeric after the first '-')
                    match = re.search(r'-(\d+)_', checkpoint_file)
                    if match:
                        mace_config_dict['seed'] = int(match.group(1))
                    with checkpoints_folder.open(checkpoint_file, 'rb') as source:                        
                        with folder.open(f'checkpoints/{checkpoint_file}', 'wb') as destination:
                            destination.write(source.read())
                elif '_epoch' in checkpoint_file:
                    with checkpoints_folder.open(checkpoint_file, 'rb') as source:                        
                        with folder.open(f'checkpoints/{checkpoint_file}', 'wb') as destination:
                            destination.write(source.read())

        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(mace_config_dict, yaml_file, default_flow_style=False)
        calcinfo = datastructures.CalcInfo()
        calcinfo.local_copy_list = [
            (
                self.inputs.finetune_model.uuid,
                self.inputs.finetune_model.filename,
                "finetune_model.dat",
            ),]
        if do_preprocess:
            calcinfo.codes_info = [codeinfo_preprocess, codeinfo, codeinfo_postprocess1, codeinfo_postprocess1b, codeinfo_postprocess2]
        else:
            calcinfo.codes_info = [codeinfo, codeinfo_postprocess1, codeinfo_postprocess1b, codeinfo_postprocess2]
        calcinfo.retrieve_list = ['*model*', 'checkpoints', 'mace.out', 'results', 'logs', '_scheduler-std*']

        return calcinfo
