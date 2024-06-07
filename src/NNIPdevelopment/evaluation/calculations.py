"""
Calculations provided by aiida_diff.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida.common import datastructures
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, StructureData, List, WorkChainNode, load_node, Int
from aiida.plugins import WorkflowFactory
import io
from contextlib import redirect_stdout
from ase.io import write
import numpy as np

# MaceWorkChain = WorkflowFactory('maceworkchain')

def dataset_list_to_txt(dataset_list):
    """Convert dataset list to xyz file."""

    dataset_txt = ''

    exclude_params = ['cell', 'symbols', 'positions', 'forces', 'stress', 'energy', 'dft_forces', 'dft_stress', 'dft_energy', 'md_forces', 'md_stress', 'md_energy']
    for config in dataset_list.get_list():
        params = [key for key in config.keys() if key not in exclude_params]
        atm = Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell'])
        atm.info = {}
        for key in params:
            atm.info[key] = config[key]
        if 'dft_stress' in config.keys():
            s = config['dft_stress']
            atm.info['dft_stress'] = f"{s[0][0]:.6f} {s[1][1]:.6f} {s[2][2]:.6f} {s[1][2]:.6f} {s[0][2]:.6f} {s[0][1]:.6f}"

        if 'dft_energy' in config.keys():
            atm.info['dft_energy'] = config['dft_energy']
        if 'dft_forces' in config.keys():
            atm.set_calculator(SinglePointCalculator(atm, forces=config['dft_forces']))
        
        with io.StringIO() as buf, redirect_stdout(buf):
            write('-', atm, format='extxyz', write_results=True, write_info=True)
            dataset_txt += buf.getvalue()


    return dataset_txt

class EvaluationCalculation(CalcJob):
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
        spec.inputs["metadata"]["options"]["parser_name"].default = "NNIPdevelopment.evaluation"

        # new ports
        # spec.input("mace_workchain", valid_type=Int, required=False, help="Mace workchain",)
        spec.input_namespace("mace_potentials", valid_type=SinglefileData, required=True, help="Mace potentials",)
        spec.input("datasetlist", valid_type=List, required=True, help="Optional list on which to compute errors.")
        spec.output("evaluated_list", valid_type=List, help="List of evaluated configurations.")
        

        spec.exit_code(300, "ERROR_MISSING_OUTPUT_FILES", message="Calculation did not produce all expected output files.",)

    
    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        codeinfo = datastructures.CodeInfo()
         
        
        # codeinfo.cmdline_params = '/leonardo/home/userexternal/dbidoggi/python_scripts/packages/dnn_potentials/test_configs_aiida.py . -a'.split()
            # file1_name=self.inputs.file1.filename, file2_name=self.inputs.file2.filename
        codeinfo.code_uuid = self.inputs.code.uuid
        # codeinfo.stdout_name = "lammps.out"
        
        calcinfo = datastructures.CalcInfo()
        calcinfo.local_copy_list = []
        # wc = load_node(19359)
        
        n_pot = 0
        for _, pot in self.inputs.mace_potentials.items():
            n_pot += 1
            calcinfo.local_copy_list.append((pot.uuid, pot.filename, f"potential_{n_pot}.dat"))

        dataset_list = self.inputs.datasetlist
        dataset_txt = dataset_list_to_txt(dataset_list)
        with folder.open("dataset.xyz", "w") as handle:
            handle.write(dataset_txt)
        codeinfo.cmdline_params = '/data/fast/35353/python_scripts/test_configs_aiida.py . -s'.split()

        
        calcinfo.codes_info = [codeinfo]
        

        calcinfo.retrieve_list = ['*']

        return calcinfo
