import yaml

def generate_lammps_md_config(temperatures=range(0,1001,100), pressures=[-5000,0,5000], steps=[10000], styles=['npt'], dt=0.001):
    """
    Generate a YAML-like configuration for a set of parameters.
    
    Parameters:
        temperatures (list): A list of temperatures.

        pressures (list): A list of pressures.

        steps (list): A list of max number of steps for the integration.

        dt (float): Timestep of simulation. Thes parameter used for thermostat parameters.
        A Nose-Hoover thermostat will not work well for arbitrary values of Tdamp. 
        If Tdamp is too small, the temperature can fluctuate wildly; if it is too large, the temperature will take a very long time to equilibrate. 
        A good choice for many models is a Tdamp of around 100 timesteps. Note that this is NOT the same as 100 time units for most units settings. 
        A simple way to ensure this, is via using an immediate variable expression accessing the thermo property ‘dt’, which is the length of the time step. 
        A Nose-Hoover barostat will not work well for arbitrary values of Pdamp. If Pdamp is too small, the pressure and volume can fluctuate wildly; 
        if it is too large, the pressure will take a very long time to equilibrate. A good choice for many models is a Pdamp of around 1000 timesteps.

        styles (list): A list of integration styles (e.g., "npt", "nvt" ...).
        
    Returns:
        str: A YAML-formatted string of the configuration.
    """
    
    config = []
    for temp in temperatures:
        for press in pressures:
            for step in steps:
                for style in styles:
                    
                    constraint = {
                        "temp": [temp, temp, 100*dt],
                        "x": [press, press, 1000*dt],
                        "y": [press, press, 1000*dt],
                        "z": [press, press, 1000*dt]           
                    }
                    md_block = {
                        "max_number_steps": step,
                        "velocity": [{"create": {"temp": temp}}],
                        "integration": {
                            "style": style,
                            "constraints": constraint                
                        }
                    }
                    config.append(md_block)

    return config
