"""
Calculations provided by aiida_diff.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, StructureData, List, FolderData, Str, Dict, Bool
import io
from contextlib import redirect_stdout
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import random
import yaml
import re
import random



# def dataset_list_to_xyz(dataset_list):
def dataset_list_to_txt(dataset_list):
    """Convert dataset list to xyz file."""

    dataset_txt = ''

    for config in dataset_list.get_list():
        atm = Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell'])
        if 'stress' in config.keys():
            s = config['stress']
        else:
            s = config['dft_stress']
        stress = [s[0][0] , s[1][1], s[2][2], s[1][2], s[0][2], s[0][1]]
        if 'energy' in config.keys() and 'forces' in config.keys():
            atm.set_calculator(SinglePointCalculator(atm, energy=config['energy'], forces=config['forces'], stress=stress))
        else:
            atm.set_calculator(SinglePointCalculator(atm, energy=config['dft_energy'], forces=config['dft_forces'], stress=stress))

        with io.StringIO() as buf, redirect_stdout(buf):
            write('-', atm, format='extxyz')
            txt_mod = buf.getvalue().replace('energy', 'dft_energy').replace('stress', 'dft_stress').replace('forces', 'dft_forces')
        if len(atm.get_chemical_symbols()) == 1:
            txt_mod = txt_mod.replace("pbc=", "config_type=IsolatedAtom pbc=")
        dataset_txt += txt_mod

    # return SinglefileData(file=io.BytesIO(dataset_txt.encode()), filename="dataset.xyz")
    return dataset_txt



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
        spec.inputs["metadata"]["options"]["parser_name"].default = "NNIPdevelopment.macetrain"
        #ADD input parameters

        # new ports
        spec.input("training_set", valid_type=List, help="Training dataset list",)
        spec.input("validation_set", valid_type=List, help="Validation dataset list",)
        spec.input("test_set", valid_type=List, help="Test dataset list",)
        spec.input("mace_config", valid_type=Dict, help="Config parameters for MACE",)
        spec.input("checkpoints", valid_type=FolderData, help="Checkpoints file", required=False)
        spec.input("preprocess_script", valid_type=SinglefileData, help="Preprocess script for parallel calculation", required=False)
        spec.input("restart", valid_type=Bool, help="Restart from a previous calculation", required=False, default=lambda:Bool(False))

        spec.output("aiida_model", valid_type=SinglefileData, help="Model file",)
        spec.output("aiida_swa_model", valid_type=SinglefileData, help="SWA Model file",)
        spec.output("aiida_compiled_model", valid_type=SinglefileData, help="Compiled Model file",)
        spec.output("aiida_swa_compiled_model", valid_type=SinglefileData, help="SWA Compiled Model file",)
        spec.output("aiida_model_lammps", valid_type=SinglefileData, help="Model lammps file",)
        spec.output("aiida_swa_model_lammps", valid_type=SinglefileData, help="SWA Model lammps file",)
        spec.output("mace_out", valid_type=SinglefileData, help="Mace output file",)
        spec.output("results", valid_type=FolderData, help="Results file",)
        spec.output("logs", valid_type=FolderData, help="Logs file",)
        spec.output("checkpoints", valid_type=FolderData, help="Checkpoints file",)
        spec.output("RMSE", valid_type=List, help="List of the checkpoints result table",)
        spec.output("results", valid_type=List, help="List of the results of log file",)
        

        spec.exit_code(300, "ERROR_MISSING_OUTPUT_FILES", message="Calculation did not produce all expected output files.",)

    @classmethod
    def get_builder_from_protocol(
        cls, **kwargs):

        builder = cls.get_builder(**kwargs)

        return builder
    
    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params =f"""--config config.yml""".split()                 

        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "mace.out"
        
        training_txt = dataset_list_to_txt(self.inputs.training_set)
        validation_txt = dataset_list_to_txt(self.inputs.validation_set)
        test_txt = dataset_list_to_txt(self.inputs.test_set)

        with folder.open('training.xyz', "w") as handle:
            handle.write(training_txt)
        with folder.open('validation.xyz', "w") as handle:
            handle.write(validation_txt)
        with folder.open('test.xyz', "w") as handle:
            handle.write(test_txt)

        # Retrieve inputs
        script = self.inputs.preprocess_script
        
        # Copy the script to the temporary folder
        script_path = folder.get_abs_path(script.filename)
        with script.open(mode='rb') as script_file:
            with open(script_path, 'wb') as temp_script_file:
                temp_script_file.write(script_file.read())
        
        mace_config_dict = self.inputs.mace_config.get_dict()
        mace_config_dict['seed'] = random.randint(0, 10000) 
        mace_config_dict['train_file'] = "processed_data/train/"   
        mace_config_dict['valid_file'] = "processed_data/val/"
        mace_config_dict['test_file'] = "processed_data/test/"    
        mace_config_dict['statistics_file'] = "processed_data/statistics.json"  
        mace_config_dict['energy_key'] = "dft_energy" 
        mace_config_dict['forces_key'] = "dft_forces" 
        mace_config_dict['stress_key'] = "dft_stress"   
        if 'checkpoints' in self.inputs:
            mace_config_dict['restart_latest'] = True

        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(mace_config_dict, yaml_file, default_flow_style=False)

        # Save the checkpoints folder
        if 'checkpoints' in self.inputs and self.inputs.restart.value==True:
            mace_config['restart_latest'] = 'true'
            checkpoints_folder = self.inputs.checkpoints
            folder.get_subfolder('checkpoints', create=True)  # Create the checkpoints directory
            for checkpoint_file in checkpoints_folder.list_object_names():

                if '_epoch' in checkpoint_file and '_swa':
                    with checkpoints_folder.open(checkpoint_file, 'rb') as source:
                        new_checkpoint_file = f"aiida_run-{str(mace_config['seed'])}_epoch-0_swa.pt"
                        with folder.open(f'checkpoints/{new_checkpoint_file}', 'wb') as destination:
                            destination.write(source.read())
                elif '_epoch' in checkpoint_file:
                    with checkpoints_folder.open(checkpoint_file, 'rb') as source:
                        new_checkpoint_file = f"aiida_run-{str(mace_config['seed'])}_epoch-0.pt"
                        with folder.open(f'checkpoints/{new_checkpoint_file}', 'wb') as destination:
                            destination.write(source.read())
                # Extract numbers from the filename using regex
                # numbers_match = re.search(r'\d+', checkpoint_file)
                # if numbers_match:
                #     original_numbers = numbers_match.group()
                #     # Replace the extracted numbers with mace_config_dict['seed'] in the filename
                #     new_checkpoint_file = checkpoint_file.replace(original_numbers, str(mace_config['seed']))
                #     with checkpoints_folder.open(checkpoint_file, 'rb') as source:
                #         with folder.open(f'checkpoints/{new_checkpoint_file}', 'wb') as destination:
                #             destination.write(source.read())
        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(mace_config, yaml_file, default_flow_style=False)
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['*model*', 'checkpoints', 'mace.out', 'results', 'logs', '_scheduler-std*']

        return calcinfo
