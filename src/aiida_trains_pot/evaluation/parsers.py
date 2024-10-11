"""
Parsers provided by aiida_diff.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, FolderData, List
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from aiida.plugins import DataFactory
import numpy as np

EvaluationCalculation = CalculationFactory("trains_pot.evaluation")
PESData = DataFactory('pesdata')

class EvaluationParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a DiffCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, EvaluationCalculation):
            raise exceptions.ParsingError("Can only parse MaceTrainCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = ['evaluated_dataset.npz']
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
        

        # add output file
        for file in files_retrieved:
            output_filename = file
            if 'evaluated_dataset' in output_filename:
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)
                with output_node.open(mode='rb') as handle:
                    # evaluated_list = np.load(handle, allow_pickle=True)
                    evaluated_list = list(np.load(handle, allow_pickle=True)['evaluated_dataset'])
                
                # self.out('evaluated_dataset', output_node)
                pse_eavaluated_list = PESData()
                pse_eavaluated_list.set_list(evaluated_list)
                self.out('evaluated_list', pse_eavaluated_list)
            #     with self.retrieved.open(output_filename, "rb") as handle:
            #         output_node = FolderData(folder=handle)
            #     self.out(output_filename, output_node)
            # elif 'logs'     
        return ExitCode(0)
