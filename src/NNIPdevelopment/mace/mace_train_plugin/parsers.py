"""
Parsers provided by aiida_diff.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, FolderData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

MaceTrainCalculation = CalculationFactory("NNIPdevelopment.macetrain")


class MaceBaseParser(Parser):
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
        if not issubclass(node.process_class, MaceTrainCalculation):
            raise exceptions.ParsingError("Can only parse MaceTrainCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = ['aiida.model']
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
        

        # add output file
        for file in files_retrieved:
            output_filename = file
            self.logger.info(f"Parsing '{output_filename}'")
            if 'checkpoint' in output_filename or 'logs' in output_filename or 'results' in output_filename:
            #     with self.retrieved.open(output_filename, "rb") as handle:
            #         output_node = FolderData(folder=handle)
            #     self.out(output_filename, output_node)
                pass
            else:
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)
                if 'model' in output_filename and not 'pt' in output_filename:
                    self.out(output_filename.replace('.','_'), output_node)
                if 'model' in output_filename and 'pt' in output_filename:
                    self.out(output_filename.replace('.pt','').replace('.','_').replace('-','_'), output_node)
                if 'mace' in output_filename:
                    self.out(output_filename.replace('.out','_out'), output_node)         
        return ExitCode(0)
