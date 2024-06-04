import re
from aiida.orm import load_node, SinglefileData
import json

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

# Example usage:
file_path = 'aiida_run-3904_train.txt'  # Replace with the actual path to your file
parsed_data = parse_log_file(file_path)

# Print the parsed data
for entry in parsed_data:
    print(entry)

# Example usage:
#node = load_node(44866) 
#parsed_data = parse_tables_from_singlefiledata(node)
#for table in parsed_data:
#    print(table)