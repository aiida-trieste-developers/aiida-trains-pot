# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_ , while_, ExitCode
from aiida import load_profile, orm
from aiida.orm import Code, Float, Str, StructureData, Int, Float, SinglefileData, Bool
from aiida.plugins import CalculationFactory
from aiida_lammps.data.potential import LammpsPotentialData
from pathlib import Path
from ase.data import atomic_numbers, atomic_masses
import numpy as np
import tempfile
import os
import io
load_profile()

LammpsCalculation = CalculationFactory('lammps.base')

class LammpsWorkChain(WorkChain):
    """WorkChain to launch LAMMPS calculations."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.expose_inputs(LammpsCalculation, namespace="lmp", exclude=('input','structure','potential'), namespace_options={'validator': None})
        spec.expose_outputs(LammpsCalculation, namespace="lmp_out")
        # spec.expose_inputs(LammpsCalculation, namespace="lmp")

        # spec.exit_code(309,"ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR", message="No safe exit code detected after 5 iterations")

        spec.input("code", valid_type=Code)
        #spec.input("temperature", valid_type=Float)
        #spec.input("dt", valid_type=Float)
        #spec.input("num_steps", valid_type=Int)
        #spec.input("pressure", valid_type=Float)
        #spec.input("parameters", valid_type=orm.Dict)
        #spec.input("settings", valid_type=orm.Dict)
        spec.input("structure", valid_type=StructureData)
        spec.input("potential", valid_type=SinglefileData)
        spec.input("parent_folder", valid_type=Str)
        spec.input("boundary", valid_type=Str, default=lambda: Str('p p f'))
        spec.input("vdw_d2", valid_type=Bool, default=lambda: Bool(False))

        spec.outline(
            cls.initialize,
            cls.run_lammps,
            while_(cls.not_converged)(
                cls.run_lammps,
            ),
            cls.finalize,
            # cls.save_files
        )

        # spec.output("coord_lmmpstrj", valid_type=SinglefileData, help="trajectory coord lammpstrj file")
        # spec.output("msd", valid_type=SinglefileData, help="msd file")
        # spec.output_namespace("rdf", valid_type=SinglefileData, dynamic=True, help="output rdf files")
        # spec.output("lammps_out", valid_type=SinglefileData, help="lammps output file")
        # spec.output("final_structure", valid_type=SinglefileData, help="final structure file")
        # spec.output("lmp_restart", valid_type=SinglefileData, help="lmp restart file")
        # spec.output("coord_atom", valid_type=SinglefileData, help="trajectory coord atom file")
        # spec.output("log_lammps", valid_type=SinglefileData, help="log lammps file")
   

    def generate_potential(self) -> LammpsPotentialData:
        """
        Generate the potential to be used in the calculation.

        Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

        :return: potential to do the calculation
        :rtype: LammpsPotentialData
        """

        potential_parameters = {
            "species": ["C"],
            "atom_style": "atomic",        
            "units": "metal",
            "extra_tags": {},
                
        }


        # Assuming you have a trained MACE model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with self.inputs.potential.open(mode='rb') as potential_handle:
                potential_content = potential_handle.read()
            tmp_file.write(potential_content)
            tmp_file_path = tmp_file.name
            
        potential = LammpsPotentialData.get_or_create(
            #source=binary_stream,
            source = Path(tmp_file_path),
            pair_style="mace",
            **potential_parameters,
        )

        os.remove(tmp_file_path)
    
        return potential

#     def input_lammps(self):
#         """Return input text."""
        
#         temp = self.inputs.temperature.value
#         dt = self.inputs.dt.value
#         num_steps = self.inputs.num_steps.value
#         structure = self.inputs.structure
#         press = self.inputs.pressure.value
#         try:
#             boundary = self.inputs.boundary.value
#         except:
#             boundary = 'p p f'

#         try:
#             vdw_d2 = self.inputs.vdw_d2.value
#         except:
#             vdw_d2 = False

#         ryd2ev = 13.605693009
#         bohr2ang = 0.52917721067

#         dftd2_c6 =[
#             4.857,    2.775,    55.853,   55.853,   108.584,
#             60.710,   42.670,   24.284,   26.018,   21.855,
#             198.087,  198.087,  374.319,  320.200,  271.980,
#             193.230,  175.885,  159.927,  374.666,  374.666,
#             374.666,  374.666,  374.666,  374.666,  374.666,
#             374.666,  374.666,  374.666,  374.666,  374.666,
#             589.405,  593.221,  567.896,  438.498,  432.600,
#             416.642,  855.833,  855.833,  855.833,  855.833,
#             855.833,  855.833,  855.833,  855.833,  855.833,
#             855.833,  855.833,  855.833,  1294.678, 1342.899,
#             1333.532, 1101.101, 1092.775, 1040.391, 10937.246,
#             7874.678, 6114.381, 4880.348, 4880.348, 4880.348,
#             4880.348, 4880.348, 4880.348, 4880.348, 4880.348,
#             4880.348, 4880.348, 4880.348, 4880.348, 4880.348,
#             4880.348, 3646.454, 2818.308, 2818.308, 2818.308,
#             2818.308, 2818.308, 2818.308, 2818.308, 1990.022,
#             1986.206, 2191.161, 2204.274, 1917.830, 1983.327,
#             1964.906
#             ]


#         dftd2_r0 = [
#             1.892, 1.912, 1.559, 2.661, 2.806,
#             2.744, 2.640, 2.536, 2.432, 2.349,
#             2.162, 2.578, 3.097, 3.243, 3.222,
#             3.180, 3.097, 3.014, 2.806, 2.785,
#             2.952, 2.952, 2.952, 2.952, 2.952,
#             2.952, 2.952, 2.952, 2.952, 2.952,
#             3.118, 3.264, 3.326, 3.347, 3.305,
#             3.264, 3.076, 3.035, 3.097, 3.097,
#             3.097, 3.097, 3.097, 3.097, 3.097,
#             3.097, 3.097, 3.097, 3.160, 3.409,
#             3.555, 3.575, 3.575, 3.555, 3.405,
#             3.330, 3.251, 3.313, 3.313, 3.313,
#             3.313, 3.313, 3.313, 3.313, 3.313,
#             3.313, 3.313, 3.313, 3.313, 3.313,
#             3.313, 3.378, 3.349, 3.349, 3.349,
#             3.349, 3.349, 3.349, 3.349, 3.322,
#             3.752, 3.673, 3.586, 3.789, 3.762,
#             3.636
#             ]


#         def vdw_au2metal(c6, r0):
#             c6 = c6*ryd2ev*bohr2ang**6
#             r0 = r0*bohr2ang
#             return c6, r0

#         def couple_vdw_params(atomic_numbers):
#             c6_1, r0_1 = vdw_au2metal(dftd2_c6[atomic_numbers[0]-1], dftd2_r0[atomic_numbers[0]-1])
#             c6_2, r0_2 = vdw_au2metal(dftd2_c6[atomic_numbers[1]-1], dftd2_r0[atomic_numbers[1]-1])
#             c6_12 = (c6_1*c6_2)**0.5
#             r0_12 = (r0_1+r0_2)
#             return c6_12, r0_12


#         atm_species = ''
#         mass_lines = ''
#         vdw_lines = ''
#         if vdw_d2:
#             vdw_lines = 'pair_style momb 20.0 0.75 20.0\n'


#         symbols = list(structure.get_symbols_set())
#         atm_nums = [atomic_numbers[symb] for symb in symbols]
#         masses = [atomic_masses[num] for num in atm_nums]

#         #        sort by mass both masses and symbols
#         masses, symbols = zip(*sorted(zip(masses, symbols)))
#         for ii in range(len(masses)):
#             atm_species += f'{symbols[ii]} '
#             mass_lines += f'mass {ii+1} {masses[ii]:>8.4f} #{symbols[ii]}\n'
#             if vdw_d2:
#                 symbol_1 = symbols[ii]
#                 for symbol_2 in symbols[ii:]:
#                     atm_numbers = [atomic_numbers[symbol_1], atomic_numbers[symbol_2]]
#                     c6_12, r0_12 = couple_vdw_params(atm_numbers)
#                     vdw_lines += f"pair_coeff {ii:<2} {symbols.index(symbol_2):<2} 0.0 1.0 1.0 {c6_12:>9.3f} {r0_12:>7.3f}   # {symbol_1:<2} {symbol_2:<2}\n"



#         input_lammps = f"""####### initialization ############
# units metal
# atom_style atomic
# atom_modify map yes #### QUESTE 2 RIGHE SOLO PER MACE ####
# newton on           #### QUESTE 2 RIGHE SOLO PER MACE ####
# boundary {boundary}

# timer timeout 23:50:00 every 100
# neigh_modify 	every 1 delay 5 check yes

# ###### atom system definition ###########
# read_data  structure.dat
# {mass_lines}

# ###### Interatomic potential ############
# pair_style mace no_domain_decomposition
# pair_coeff * * potential.dat {atm_species}
# {vdw_lines}
# variable settemp equal {temp}
# velocity  all create {temp} {np.random.randint(1,1000)}

# ############### time step ############
# timestep {dt} 

# ########## command for simulation #######
# # compute press all pressure thermo_temp virial
# #compute_modify thermo_press pressure thermo_temp
# fix mysim all npt temp {temp} {temp} $(100.0*dt) x {press} {press} $(1000.0*dt) y {press} {press} $(1000.0*dt)
# # fix press all 
# compute myRDF1 all rdf 1000 1 1
# # compute myRDF2 all rdf 1000 1 2
# # compute myRDF3 all rdf 1000 2 2
# compute myADF all adf 180 1 1 1 1 5 1 5 #&
#                         #   1 1 2 1 5 1 5 &
#                         #   1 2 2 1 5 1 5 &
#                         #   2 2 2 1 5 1 5 &
#                         #   2 2 1 1 5 1 5 &
#                         #   2 1 1 1 5 1 5
# compute mymsd all msd com yes
# fix 1 all ave/time 100 1 100 c_myRDF1[*] file rdf11.rdf mode vector
# # fix 2 all ave/time 100 1 100 c_myRDF2[*] file rdf12.rdf mode vector
# # fix 3 all ave/time 100 1 100 c_myRDF3[*] file rdf22.rdf mode vector
# fix 4 all ave/time 1 1 100 c_mymsd[*] file msd.msd
# fix 5 all ave/time 100 1 100 c_myADF[*] file adf.adf mode vector
# #
# thermo		20
# thermo_style custom step time temp lx ly lz vol pxx pyy pzz pe ke etotal press spcpu cpuremain
# dump mydump all custom 100 coord.atom type id x y z fx fy fz
# #dump mydump2 all traj 100 coord.traj
# dump mydump3 all atom 100 coord.lammpstrj
# #dump out all yaml 100 dump.yaml id type x y z vx vy vz ix iy iz fx fy fz
# #dump mydump2 all custom 1000 force.atom type id fx fy fz

# ############## run simulation ##############
# run		{num_steps}
# write_data lmp.data
# write_restart lmp_restart.rest"""
        
#         string_data = io.BytesIO(input_lammps.encode())
#         input_file = SinglefileData(file=string_data, filename="input.in")
#         return input_file


    @classmethod
    def get_builder_from_protocol(
        cls, **kwargs):
        
        lmp = LammpsCalculation.get_builder_from_protocol(**kwargs)

        builder = cls.get_builder()
        # builder.lmp = lmp

        return builder

    def initialize(self):
        """Initialize."""
        self.iteration = 0        

    def run_lammps(self):
        """Run Lammps calculations."""
        self.iteration += 1
        #input_file = self.input_lammps()

        future = self.submit(LammpsCalculation,
                             code=self.inputs.code,
                             #settings = self.inputs.settings,
                             structure=self.inputs.structure,
                             #parameters = self.inputs.parameters,
                             potential=self.generate_potential(),
                             #input=input_file,
                             **self.exposed_inputs(LammpsCalculation, namespace="lmp"))

        self.report(f'launched lammps calculation <{future.pk}>')
        self.to_context(lammps_calculations=append_(future))

    def not_converged(self):
        """Check if calculuation ended without errors."""
        # if self.ctx.lammps_calculations[-1].
        if self.ctx.lammps_calculations[-1].exit_status != 0 and self.iteration < 5:
            return True
        else:
            return False    


    def finalize(self):
        """Finalize."""
        # self.report(f'iteration {self.iteration}, exit status {self.ctx.lammps_calculations[-1].exit_status}')
        if self.ctx.lammps_calculations[-1].exit_status == 0:
            self.out_many(self.exposed_outputs(self.ctx.lammps_calculations[-1], LammpsCalculation, namespace="lmp_out"))
        else:
            return ExitCode(309, 'Lammps calculation did not end correctly after 5 iterations')
            # self.out_many(self.exposed_outputs(self.ctx.lammps_calculations[-1], LammpsCalculation, namespace="lmp_out"))
            # self.exit_codes.ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR
        # for ii, val in enumerate(self.ctx.lammps_calculations):
        #     self.out(f'rdf', val.outputs.rdf)
        #     self.out(f'coord_lmmpstrj', val.outputs.coord_lmmpstrj)
        #     self.out(f'msd', val.outputs.msd)
        #     self.out(f'lammps_out', val.outputs.lammps_out)
        #     self.out(f'final_structure', val.outputs.final_structure)
        #     self.out(f'lmp_restart', val.outputs.lmp_restart)
        #     self.out(f'coord_atom', val.outputs.coord_atom)
        #     self.out(f'log_lammps', val.outputs.log_lammps)



    def save_files(self):
        """Create folder and save files."""
        folder = f'{self.inputs.parent_folder.value}/Data/LAMMPS_pot{self.inputs.potential.pk}_str{self.inputs.structure.pk}_DT{self.inputs.dt.value}_N{self.inputs.num_steps.value}_P{self.inputs.pressure.value}_T{self.inputs.temperature.value}'
        os.makedirs(folder, exist_ok=True)
        retrived = self.ctx.lammps_calculations[0].get_retrieved_node()


        for r in retrived.list_objects():
            self.report(r.name)
            with retrived.open(r.name, 'rb') as handle:
                with open(f'{folder}/{r.name}', "w") as f:
                    try:
                        f.write(handle.read().decode('utf-8'))
                    except:
                        pass

