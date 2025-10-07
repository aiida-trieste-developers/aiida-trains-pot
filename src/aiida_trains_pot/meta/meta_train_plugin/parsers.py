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
import csv


MetaTrainCalculation = CalculationFactory("trains_pot.metatrain")

def parse_tables_from_singlefiledata(node):
    """
    Parse a CSV training log from a SinglefileData node and convert it into
    an AiiDA-compatible list of dictionaries with RMSE-style fields.

    Expected CSV columns:
        Epoch,training energy RMSE (per atom),training forces RMSE,
        validation energy RMSE (per atom),validation forces RMSE, ...

    Returns:
        list[dict]: Each dictionary contains epoch + Train_default / Valid_default entries.
    """
    if not isinstance(node, SinglefileData):
        raise TypeError(f"Node {node} is not a SinglefileData node.")

    parsed_list = []

    with node.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("Epoch"):
                continue

            try:
                epoch = int(float(row["Epoch"]))
            except (ValueError, TypeError):
                continue

            # Parse training metrics
            train_data = {}
            valid_data = {}

            # Try to collect all relevant columns if present
            if "training energy RMSE (per atom)" in row:
                train_data["RMSE_E/meV/atom"] = float(row["training energy RMSE (per atom)"]) * 1000 if float(row["training energy RMSE (per atom)"]) < 10 else float(row["training energy RMSE (per atom)"])
            if "training forces RMSE" in row:
                train_data["RMSE_F/meV/A"] = float(row["training forces RMSE"])
            if "training energy MAE (per atom)" in row:
                train_data["Relative_F_RMSE_%"] = float(row["training energy MAE (per atom)"])

            if "validation energy RMSE (per atom)" in row:
                valid_data["RMSE_E/meV/atom"] = float(row["validation energy RMSE (per atom)"])
            if "validation forces RMSE" in row:
                valid_data["RMSE_F/meV/A"] = float(row["validation forces RMSE"])
            if "validation energy MAE (per atom)" in row:
                valid_data["Relative_F_RMSE_%"] = float(row["validation energy MAE (per atom)"])

        parsed_entry = {"epoch": epoch}
        if train_data:
            parsed_entry["Train_default"] = train_data
        if valid_data:
            parsed_entry["Valid_default"] = valid_data

        parsed_list.append(parsed_entry)

    return parsed_list
    """ Parses tables from a SinglefileData node in AiiDA and returns the data as dictionaries,
        including the number of epochs.

        Args:
        node_uuid (str): The UUID of the SinglefileData node containing the tables.
    
        Returns:
        list of dict: A list containing dictionaries for each parsed table with epoch information.
        
        
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

        return parsed_data"  """

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
    start_pattern = re.compile(r'Starting training')
    
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
    start_pattern = re.compile(r'Training finished!')
    
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

class MetaBaseParser(Parser):
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
        if not issubclass(node.process_class, MetaTrainCalculation):
            raise exceptions.ParsingError("Can only parse MetaTrainCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = ['model.pt']
        
        # add error of out of walltime
        if 'meta.out' in self.retrieved.list_object_names():
            with self.retrieved.open('meta.out', "rb") as handle:
                output_node = SinglefileData(file=handle) 
                if (len(parse_start_from_singlefiledata(output_node)) > 0) and (len(parse_complete_from_singlefiledata(output_node)) == 0):
                    return self.exit_codes.ERROR_OUT_OF_WALLTIME

        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        for output_filename in files_retrieved:
            self.logger.info(f"Parsing '{output_filename}'")

            # Case 1: It’s an 'outputs' folder
            if output_filename.startswith('outputs'):
                # Recursively walk inside outputs directory
                def walk_folder(base_path):
                    """Yield all file paths recursively from retrieved folder."""
                    try:
                        entries = self.retrieved.list_object_names(base_path)
                    except (OSError, FileNotFoundError):
                        # It's a file, not a directory
                        yield base_path
                        return

                    for name in entries:
                        full_path = os.path.join(base_path, name)
                        yield from walk_folder(full_path)

                for file_path in walk_folder(output_filename):
                    if file_path.endswith("train.csv"):
                        self.logger.info(f"Found train.csv at: {file_path}")
                        with self.retrieved.open(file_path, "rb") as handle:
                            output_node = SinglefileData(file=handle)
                        self.out("RMSE", List(parse_tables_from_singlefiledata(output_node)))

            # Case 2: Other single output files
            else:
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)

                # Simplify model detection logic
                if "model.pt" in output_filename:
                    self.out("model_stage2_lammps", output_node)
                #elif "model.pt" in output_filename:
                #    self.out("model_stage2_ase", output_node)
                #elif "model.pt" in output_filename:
                #    self.out("model_stage2_pytorch", output_node)

                #elif 'model.pt' in output_filename:
                #    self.out("model_stage1_lammps", output_node)
                #elif 'model.pt' in output_filename:
                #   self.out("model_stage1_ase", output_node)
                #elif 'model.pt' in output_filename:
                #    self.out("model_stage1_pytorch", output_node)                  
                
                elif output_filename.endswith("model.ckpt"):
                    folder_node = FolderData()
                    with self.retrieved.open(output_filename, "rb") as handle:
                        folder_node.put_object_from_filelike(handle, output_filename)
                    self.out("checkpoints", folder_node)
                elif output_filename.endswith(".out"):
                    self.out(output_filename.replace('.out', '_out'), output_node)


        """ # add output file
        for file in files_retrieved:
            output_filename = file
            self.logger.info(f"Parsing '{output_filename}'")         
            if 'outputs' in output_filename:
                if 'outputs' in output_filename:
                    folder_contents = self.retrieved.list_object_names(output_filename)
                    for file_in_folder in folder_contents:
                        folder_path = os.path.join(output_filename, file_in_folder)  
                        folder_contents_last = self.retrieved.list_object_names(folder_path) 

                           
                    for file_in_folder in folder_path:   
                        folder_path = os.path.join(folder_path, file_in_folder)                     
                        folder_contents_last = self.retrieved.list_object_names(folder_path)

                   
                    for file_in_folder in folder_path:                        
                        if 'train.csv' in file_in_folder:
                            file_path = os.path.join( folder_path, output_filename)
                            with self.retrieved.open(file_path, "rb") as handle:  
                                output_node = SinglefileData(file=handle)     
                                self.out("RMSE",  List(parse_tables_from_singlefiledata(output_node)))      

            if output_filename.startswith('outputs'):
                # Recursively walk inside outputs directory
                def walk_folder(base_path):

                    for name in self.retrieved.list_object_names(base_path):
                        full_path = os.path.join(base_path, name)
                        if self.retrieved.object_exists(full_path):
                            # Check if it’s a directory
                            try:
                                sub_files = self.retrieved.list_object_names(full_path)
                                yield from walk_folder(full_path)  # recurse
                            except (OSError, FileNotFoundError):
                                yield full_path  # it's a file

                for file_path in walk_folder(output_filename):
                    if file_path.endswith("train.csv"):
                        with self.retrieved.open(file_path, "rb") as handle:
                            output_node = SinglefileData(file=handle)
                        self.out("RMSE", List(parse_tables_from_singlefiledata(output_node)))                          
                
            else:
                with self.retrieved.open(output_filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)

                if "model.pt" in output_filename:
                    self.out("model_stage2_lammps", output_node)
                elif "model.pt" in output_filename:
                    self.out("model_stage2_ase", output_node)
                elif "model.pt" in output_filename:
                    self.out("model_stage2_pytorch", output_node)

                elif 'model.pt' in output_filename:
                    self.out("model_stage1_lammps", output_node)
                elif 'model.pt' in output_filename:
                    self.out("model_stage1_ase", output_node)
                elif 'model.pt' in output_filename:
                    self.out("model_stage1_pytorch", output_node)
                    
                elif 'model.ckpt' in output_filename:
                    self.out("checkpoints", output_node)
                    
                elif 'meta' in output_filename:
                    self.out(output_filename.replace('.out','_out'), output_node) """
                             
        return ExitCode(0)
