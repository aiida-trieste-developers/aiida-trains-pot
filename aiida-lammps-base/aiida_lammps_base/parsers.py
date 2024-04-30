"""
Parsers provided by aiida_diff.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

LammpsBaseCalculation = CalculationFactory("lammps_base")


class LammpsBaseParser(Parser):
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
        if not issubclass(node.process_class, LammpsBaseCalculation):
            raise exceptions.ParsingError("Can only parse DiffCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = ['coord.lammpstrj']
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
        

        output_rdf = {}
        # add output file
        for file in files_retrieved:
            output_filename = file
            self.logger.info(f"Parsing '{output_filename}'")
            with self.retrieved.open(output_filename, "rb") as handle:
                output_node = SinglefileData(file=handle)
            if 'coord' in output_filename or 'lammps' in output_filename:
                self.out(output_filename.replace('.','_'), output_node)
            if 'rdf' in output_filename:
                output_rdf[output_filename.replace('.rdf','')] = output_node
            if 'msd' in output_filename:
                self.out("msd", output_node)
            if 'adf' in output_filename:
                self.out("adf", output_node)
            # if 'scheduler' in output_filename:
            #     self.out(output_filename.replace('.txt',''), output_node)
            if 'lmp_restart' in output_filename:
                self.out("lmp_restart", output_node)
            if 'lmp.data' in output_filename:
                self.out("final_structure", output_node)            
        self.out('rdf', output_rdf)
        return ExitCode(0)
