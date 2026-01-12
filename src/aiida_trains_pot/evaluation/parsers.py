"""Parsers provided by aiida_diff.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

import numpy as np

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory, DataFactory

EvaluationCalculation = CalculationFactory("trains_pot.evaluation")
PESData = DataFactory("pesdata")
ArrayData = DataFactory("core.array")


class EvaluationParser(Parser):
    """Parser class for parsing output of calculation."""

    def __init__(self, node):
        """Initialize Parser instance.

        Checks that the ProcessNode being passed was produced by a DiffCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, EvaluationCalculation):
            raise exceptions.ParsingError("Can only parse MaceTrainCalculation")

    def parse(self, **kwargs):
        """Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        if True not in ["evaluated.npz" in file for file in files_retrieved]:
            self.logger.error("No evaluated dataset found")
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        datasets = {}
        rmse = {}
        parity = {}
        # add output file
        for file in files_retrieved:
            output_filename = file

            if "dataset" in output_filename and "evaluated" in output_filename:
                output_name = output_filename.replace("_evaluated.npz", "").replace("dataset_", "")
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)
                with output_node.open(mode="rb") as handle:
                    evaluated_list = list(np.load(handle, allow_pickle=True)["evaluated_dataset"])
                    datasets[output_name] = PESData(evaluated_list)

            if "dataset" in output_filename and "rmse" in output_filename:
                output_name = output_filename.replace("_rmse.npz", "").replace("dataset_", "")
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)
                with output_node.open(mode="rb") as handle:
                    rmse[output_name] = Dict(np.load(handle, allow_pickle=True)["rmse"].tolist())

            if "dataset" in output_filename and "parity" in output_filename:
                output_name = output_filename.replace("_parity.npz", "").replace("dataset_", "")
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)
                with output_node.open(mode="rb") as handle:
                    parity[output_name] = ArrayData(np.load(handle, allow_pickle=True)["parity"].tolist())

        self.out("evaluated_datasets", datasets)
        self.out("rmse", rmse)
        self.out("parity_data", parity)

        return ExitCode(0)
