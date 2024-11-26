import yaml

def generate_lammps_md_config(temperatures, steps, constraints, styles):
    """
    Generate a YAML-like configuration for a set of parameters.
    
    Parameters:
        temperatures (list): A list of temperatures for the velocity create step.
        steps (list): A list of max number of steps for the integration.
        constraints (dict): Constraints for the integration style.
        styles (list): A list of integration styles (e.g., "npt", "nvt" ...).
        
    Returns:
        str: A YAML-formatted string of the configuration.
    """
    if not (len(temperatures) == len(steps) == len(constraints) == len(styles)):
        raise ValueError("The lengths of temperatures, steps, and styles must match.")
    
    config = []
    for temp, step, constraint, style in zip(temperatures, steps, constraints, styles):
        integration_block = {
            "integration": {
                "style": style,
                "constraints": constraint,
                "max_number_steps": step,
                "velocity": [{"create": {"temp": temp}}]
            }
        }
        config.append(integration_block)
    
    return config
