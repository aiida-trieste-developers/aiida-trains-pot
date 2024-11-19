"""
Parsers provided by aiida_diff.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, FolderData, List, Dict
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
import re
import json
import os


MaceTrainCalculation = CalculationFactory("trains_pot.macetrain")

def parse_tables_from_singlefiledata(node):
    """
    Parses tables from a SinglefileData node in AiiDA and returns the data as dictionaries,
    including the number of epochs.

    Args:
    node_uuid (str): The UUID of the SinglefileData node containing the tables.
 
    Returns:
    list of dict: A list containing dictionaries for each parsed table with epoch information.
    """
    
    if not isinstance(node, SinglefileData):
        raise TypeError(f'Node {node} is not a SinglefileData node.')

    # List to store the parsed data
    parsed_data = []

    # Read the file content
    with node.open() as file:
        lines = file.readlines()

    # Regular expression patterns
    epoch_pattern = re.compile(r'Loading checkpoint: .*_epoch-(\d+).*')
    saving_model_pattern = re.compile(r'Saving model to ')
    table_row_pattern = re.compile(r'\|\s+(\w+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|')

    current_epoch = None
    table_data = {}

    for line in lines:
        # Check for epoch information
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
    

        # Parse table rows       
        row_match = table_row_pattern.search(line)
        if row_match:
            config_type = row_match.group(1)
            rmse_e = float(row_match.group(2))
            rmse_f = float(row_match.group(3))
            relative_f_rmse = float(row_match.group(4))
            table_data[config_type.capitalize()] = {
                "RMSE_E/meV/atom": rmse_e,
                "RMSE_F/meV/A": rmse_f,
                "Relative_F_RMSE_%": relative_f_rmse
            }

        if saving_model_pattern.search(line):
            table_data['epoch'] = current_epoch
            parsed_data.append(table_data)
            table_data = {}

    return parsed_data

def parse_start_from_singlefiledata(node):
    """
    Parses tables from a SinglefileData node in AiiDA and returns the list with start training line 

    Args:
    node_uuid (str): The UUID of the SinglefileData node containing the tables.
 
    Returns:
    list: A list of string
    """
    
    if not isinstance(node, SinglefileData):
        raise TypeError(f'Node {node} is not a SinglefileData node.')

    # List to store the parsed data
    parsed_data = []

    # Read the file content
    with node.open() as file:
        lines = file.readlines()

    # Regular expression patterns
    start_pattern = re.compile(r'Started training')
    
    for line in lines:
        # Check for epoch information
        epoch_match = start_pattern.search(line)
        
        if epoch_match:
            parsed_data.append(line)
            
    return parsed_data

def parse_complete_from_singlefiledata(node):
    """
    Parses tables from a SinglefileData node in AiiDA and returns the list with complete training line 

    Args:
    node_uuid (str): The UUID of the SinglefileData node containing the tables.
 
    Returns:
    list: A list of string
    """
    
    if not isinstance(node, SinglefileData):
        raise TypeError(f'Node {node} is not a SinglefileData node.')

    # List to store the parsed data
    parsed_data = []

    # Read the file content
    with node.open() as file:
        lines = file.readlines()

    # Regular expression patterns
    start_pattern = re.compile(r'Training complete')
    
    for line in lines:
        # Check for epoch information
        epoch_match = start_pattern.search(line)
        
        if epoch_match:
            parsed_data.append(line)
            
    return parsed_data

def parse_log_file(node):
    """
    Parses a log file containing JSON-like entries and returns a list of parsed JSON objects
    that match the required format.

    Args:
    file_path (str): The path to the log file.

    Returns:
    list: A list of parsed JSON objects.
    """
    if not isinstance(node, SinglefileData):
        raise TypeError(f'Node {node} is not a SinglefileData node.')

    # Define the required keys
    required_keys = {
        "loss", "mae_e", "mae_e_per_atom", "rmse_e", "rmse_e_per_atom",
        "q95_e", "mae_f", "rel_mae_f", "rmse_f", "rel_rmse_f", "q95_f",
        "mae_stress", "rmse_stress", "rmse_stress_per_atom", "q95_stress",
        "mae_virials", "rmse_virials", "rmse_virials_per_atom", "q95_virials",
        "time", "mode", "epoch"
    }

    parsed_data = []
    with node.open() as file:
        for line in file:
            # Parse the JSON-like entry
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip lines that aren't valid JSON
            
            # Check if the entry contains all required keys
            if required_keys.issubset(entry.keys()):
                # Add the entry to the list
                parsed_data.append(entry)
    
    return parsed_data

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
        
        # add error of out of walltime
        if 'mace.out' in self.retrieved.list_object_names():
            with self.retrieved.open('mace.out', "rb") as handle:
                output_node = SinglefileData(file=handle) 
                if (len(parse_start_from_singlefiledata(output_node)) > 0) and (len(parse_complete_from_singlefiledata(output_node)) == 0):
                    return self.exit_codes.ERROR_OUT_OF_WALLTIME

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
                if 'results' in output_filename:
                    folder_contents = self.retrieved.list_object_names(output_filename)
                    for file_in_folder in folder_contents:
                        file_path = os.path.join(output_filename, file_in_folder)
                        with self.retrieved.open(file_path, "rb") as handle:                                           
                            parsed_results = parse_log_file(SinglefileData(file=handle))
                            self.out("results", List(parsed_results))    
                if 'logs' in output_filename:
                    folder_data = FolderData()
                    folder_contents = self.retrieved.list_object_names(output_filename)
                    for file_in_folder in folder_contents:
                        file_path = os.path.join(output_filename, file_in_folder)
                        with self.retrieved.open(file_path, "rb") as handle:
                            folder_data.put_object_from_filelike(handle, file_in_folder)
                    self.out(output_filename, folder_data)
                if 'checkpoint' in output_filename:
                    #with self.retrieved.open(output_filename, "rb") as handle:
                    #    output_node = FolderData(folder=handle)
                    #self.out(output_filename, output_node)  
                    folder_data = FolderData()
                    folder_contents = self.retrieved.list_object_names(output_filename)
                    for file_in_folder in folder_contents:
                        file_path = os.path.join(output_filename, file_in_folder)
                        with self.retrieved.open(file_path, "rb") as handle:
                            folder_data.put_object_from_filelike(handle, file_in_folder)
                    self.out(output_filename, folder_data)       
            else:
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)

                if "aiida_swa.model-lammps" in output_filename or "aiida_stagetwo.model-lammps" in output_filename:
                    self.out("model_stage2_lammps", output_node)
                elif "aiida_swa_compiled.model" in output_filename or "aiida_stagetwo_compiled.model" in output_filename:
                    self.out("model_stage2_ase", output_node)
                elif "aiida_swa.model" in output_filename or "aiida_stagetwo.model" in output_filename:
                    self.out("model_stage2_pytorch", output_node)

                elif 'aiida.model-lammps' in output_filename:
                    self.out("model_stage1_lammps", output_node)
                elif 'aiida_compiled.model' in output_filename:
                    self.out("model_stage1_ase", output_node)
                elif 'aiida.model' in output_filename:
                    self.out("model_stage1_pytorch", output_node)
                    
                elif 'mace' in output_filename:
                    self.out(output_filename.replace('.out','_out'), output_node)
                    self.out("RMSE", List(parse_tables_from_singlefiledata(output_node)))         
        return ExitCode(0)
