"""
Calculations provided by aiida_diff.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, StructureData, List, FolderData, Str, Dict
import io
from contextlib import redirect_stdout
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import yaml




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
        #spec.input("params.default_dtype", valid_type=Str, help="set default dtype", required=False,)

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
        # codeinfo.cmdline_params =f"""--name=aiida
        # --seed={random.randint(0, 10000)}
        # --train_file=training.xyz
        # --valid_file=validation.xyz
        # --test_file=test.xyz
        # --default_dtype={self.inputs.params.default_dtype.value}
        # --model=MACE
        # --compute_stress=True
        # --energy_key=dft_energy
        # --forces_key=dft_forces
        # --stress_key=dft_stress
        # --stress_weight=100
        # --hidden_irreps=128x0e+128x1o
        # --r_max=10.4
        # --batch_size=1
        # --max_num_epochs=20
        # --swa
        # --start_swa=5
        # --ema
        # --ema_decay=0.99
        # --amsgrad
        # --patience=10
        # --device=cuda
        # --loss=universal
        # --error_table=PerAtomRMSEstressvirials
        # --keep_isolated_atoms=True
        # --save_cpu""".split()
        # --E0s=average
            # file1_name=self.inputs.file1.filename, file2_name=self.inputs.file2.filename
            

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

        mace_config_dict = self.inputs.mace_config.get_dict()
        with folder.open('config.yml', 'w') as yaml_file:
            yaml.dump(mace_config_dict, yaml_file, default_flow_style=False)

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['*model*', 'checkpoints', 'mace.out', 'results', 'logs', '_scheduler-std*']

        return calcinfo
