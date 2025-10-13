from ase.data import atomic_numbers, atomic_masses

ryd2ev      = 13.605693009
bohr2ang    = 0.52917721067

dftd2_c6 =[
    4.857,    2.775,    55.853,   55.853,   108.584,
    60.710,   42.670,   24.284,   26.018,   21.855,
    198.087,  198.087,  374.319,  320.200,  271.980,
    193.230,  175.885,  159.927,  374.666,  374.666,
    374.666,  374.666,  374.666,  374.666,  374.666,
    374.666,  374.666,  374.666,  374.666,  374.666,
    589.405,  593.221,  567.896,  438.498,  432.600,
    416.642,  855.833,  855.833,  855.833,  855.833,
    855.833,  855.833,  855.833,  855.833,  855.833,
    855.833,  855.833,  855.833,  1294.678, 1342.899,
    1333.532, 1101.101, 1092.775, 1040.391, 10937.246,
    7874.678, 6114.381, 4880.348, 4880.348, 4880.348,
    4880.348, 4880.348, 4880.348, 4880.348, 4880.348,
    4880.348, 4880.348, 4880.348, 4880.348, 4880.348,
    4880.348, 3646.454, 2818.308, 2818.308, 2818.308,
    2818.308, 2818.308, 2818.308, 2818.308, 1990.022,
    1986.206, 2191.161, 2204.274, 1917.830, 1983.327,
    1964.906
    ]


dftd2_r0 = [
    1.892, 1.912, 1.559, 2.661, 2.806,
    2.744, 2.640, 2.536, 2.432, 2.349,
    2.162, 2.578, 3.097, 3.243, 3.222,
    3.180, 3.097, 3.014, 2.806, 2.785,
    2.952, 2.952, 2.952, 2.952, 2.952,
    2.952, 2.952, 2.952, 2.952, 2.952,
    3.118, 3.264, 3.326, 3.347, 3.305,
    3.264, 3.076, 3.035, 3.097, 3.097,
    3.097, 3.097, 3.097, 3.097, 3.097,
    3.097, 3.097, 3.097, 3.160, 3.409,
    3.555, 3.575, 3.575, 3.555, 3.405,
    3.330, 3.251, 3.313, 3.313, 3.313,
    3.313, 3.313, 3.313, 3.313, 3.313,
    3.313, 3.313, 3.313, 3.313, 3.313,
    3.313, 3.378, 3.349, 3.349, 3.349,
    3.349, 3.349, 3.349, 3.349, 3.322,
    3.752, 3.673, 3.586, 3.789, 3.762,
    3.636
    ]

def vdw_au2metal(c6, r0):
    """
    Convert DFT-D2 parameters from atomic units to metal units.
    Args:
        c6 (float): The C6 parameter in atomic units.
        r0 (float): The R0 parameter in atomic units.
    Returns:
        tuple: A tuple containing the converted C6 and R0 parameters.
    """
    c6 = c6*ryd2ev*bohr2ang**6
    r0 = r0*bohr2ang
    return c6, r0

def couple_vdw_params(atomic_numbers):
    """
    Couple the DFT-D2 parameters for two atomic species.
    Args:
        atomic_numbers (list): A list of two atomic numbers.
    Returns:
        tuple: A tuple containing the coupled C6 and R0 parameters.
    """
    c6_1, r0_1 = vdw_au2metal(dftd2_c6[atomic_numbers[0]-1], dftd2_r0[atomic_numbers[0]-1])
    c6_2, r0_2 = vdw_au2metal(dftd2_c6[atomic_numbers[1]-1], dftd2_r0[atomic_numbers[1]-1])
    c6_12 = (c6_1*c6_2)**0.5
    r0_12 = (r0_1+r0_2)
    return c6_12, r0_12      


def get_dftd2_pair_coeffs(structure) -> list:
    """
    Get the DFT-D2 parameters for momb pair style of LAMMPS for a given structure.
    Args:
        structure (StructureData): The structure for which to get the DFT-D2 parameters.
    Returns:
        list: A list of strings containing the DFT-D2 parameters.
    """
    symbols = list(structure.get_symbols_set())
    atm_nums = [atomic_numbers[symb] for symb in symbols]
    masses = [atomic_masses[num] for num in atm_nums]

    vdw_coeffs = []
    masses, symbols = zip(*sorted(zip(masses, symbols)))
    for ii in range(len(masses)):
        symbol_1 = symbols[ii]
        for symbol_2 in symbols[ii:]:
            atm_numbers = [atomic_numbers[symbol_1], atomic_numbers[symbol_2]]
            c6_12, r0_12 = couple_vdw_params(atm_numbers)
            vdw_coeffs.append(f"{ii+1:<2} {symbols.index(symbol_2)+1:<2} momb 0.0 1.0 1.0 {c6_12:>9.3f} {r0_12:>7.3f}   # {symbol_1:<2} {symbol_2:<2}")
    return vdw_coeffs

def get_mace_pair_coeff(structure, hybrid=False) -> str:
    """
    Get the MACE pair coefficient for a given structure.
    Args:
        structure (StructureData): The structure for which to get the MACE pair coefficient.
        hybrid (bool): Whether is used hybrid/overlay pair style or not.
    Returns:
        str: The MACE pair coefficient.
    """
    if hybrid:
        return "* * mace potential.dat " + " ".join(structure.get_symbols_set())
    return "* * potential.dat " + " ".join(structure.get_symbols_set())

def get_meta_pair_coeff(structure, hybrid=False) -> str:
    """
    Get the METATrain pair coefficient for a given structure.
    Args:
        structure (StructureData): The structure for which to get the METATrain pair coefficient.
        hybrid (bool): Whether is used hybrid/overlay pair style or not.
    Returns:
        str: The METATrain pair coefficient.
    """
    #if hybrid:
    #    return "* * metatomic potential.dat " + " ".join(structure.get_symbols_set())
    symbols = sorted(structure.get_symbols_set())
    numbers = [str(atomic_numbers[s]) for s in symbols]
    return "* * " + " ".join(numbers)
