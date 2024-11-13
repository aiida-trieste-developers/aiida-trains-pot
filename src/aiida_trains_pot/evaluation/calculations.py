"""
Calculations provided by aiida_diff.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida.common import datastructures
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, StructureData, List, WorkChainNode, load_node, Int, Dict
from aiida.plugins import WorkflowFactory
from aiida.plugins import DataFactory
import numpy as np

# MaceWorkChain = WorkflowFactory('maceworkchain')
PESData = DataFactory('pesdata')


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
        spec.inputs["metadata"]["options"]["parser_name"].default = "trains_pot.evaluation"

        # new ports
        spec.input_namespace("mace_potentials", valid_type=SinglefileData, required=True, help="Mace potentials",)
        spec.input("datasetlist", valid_type=PESData, required=True, help="Optional list on which to compute errors.")
        spec.output("evaluated_list", valid_type=PESData, help="List of evaluated configurations.")
        spec.output("rmse", valid_type=Dict, help="Root mean square errors between DFT and DNN quantities for the different sets and potentials.")

        spec.exit_code(300, "ERROR_MISSING_OUTPUT_FILES", message="Calculation did not produce all expected output files.",)

    
    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = 'evaluation.out'

        calcinfo = datastructures.CalcInfo()
        calcinfo.local_copy_list = []
        
        
        n_pot = 0
        for _, pot in self.inputs.mace_potentials.items():
            n_pot += 1
            calcinfo.local_copy_list.append((pot.uuid, pot.filename, f"potential_{n_pot}.dat"))

        dataset_list = self.inputs.datasetlist
        dataset_txt = dataset_list.get_txt()
        with folder.open("dataset.xyz", "w") as handle:
            handle.write(dataset_txt)
        
        calcinfo.codes_info = [codeinfo]
        

        calcinfo.retrieve_list = ['*']

        return calcinfo
