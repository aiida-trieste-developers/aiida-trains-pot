import re
from aiida.orm import load_node
from aiida.orm.nodes.data.singlefile import SinglefileData

def parse_table_from_singlefiledata(node_uuid):
    """
    Parses a table from a SinglefileData node in AiiDA and returns the data as a dictionary.
    
    Args:
    node_uuid (str): The UUID of the SinglefileData node containing the table.
    
    Returns:
    dict: A dictionary containing the parsed table data.
    """
    # Load the SinglefileData node
    node = load_node(node_uuid)
    if not isinstance(node, SinglefileData):
        raise TypeError(f'Node {node_uuid} is not a SinglefileData node.')

    # Dictionary to store the parsed data
    parsed_data = {}

    # Read the file content
    with node.open() as file:
        lines = file.readlines()

    # Regular expression pattern to match table rows
    pattern = re.compile(r'\|\s+(\w+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|')

    # Parse the lines and populate the dictionary
    for line in lines:
        match = pattern.search(line)
        if match:
            config_type = match.group(1)
            rmse_e = float(match.group(2))
            rmse_f = float(match.group(3))
            relative_f_rmse = float(match.group(4))
            parsed_data[config_type.capitalize()] = {
                "RMSE_E/meV/atom": rmse_e,
                "RMSE_F/meV/A": rmse_f,
                "Relative_F_RMSE_%": relative_f_rmse
            }

    return parsed_data

# Example usage
node_uuid = 44866
table_data = parse_table_from_singlefiledata(node_uuid)
print(table_data)