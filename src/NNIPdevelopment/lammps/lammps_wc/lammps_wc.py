# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_ , while_, ExitCode
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, Float, SinglefileData
from aiida.plugins import CalculationFactory
import numpy as np
import os
import io
load_profile()

LammpsCalculation = CalculationFactory('NNIPdevelopment.lammpsmd')

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
        spec.input("temperature", valid_type=Float)
        spec.input("dt", valid_type=Float)
        spec.input("num_steps", valid_type=Int)
        spec.input("pressure", valid_type=Float)
        spec.input("structure", valid_type=StructureData)
        spec.input("potential", valid_type=SinglefileData)
        spec.input("parent_folder", valid_type=Str)

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
   


    def input_lammps(self):
        """Return input text."""
        
        temp = self.inputs.temperature.value
        dt = self.inputs.dt.value
        num_steps = self.inputs.num_steps.value
        structure = self.inputs.structure
        press = self.inputs.pressure.value

        structure_ase = structure.get_ase()
        masses = []
        symbols = []
        atm_species = ''
        mass_lines = ''
        num_species = 0

        symbol = structure_ase.get_chemical_symbols()
        for ii, mass in enumerate(structure_ase.get_masses()):
            if mass not in masses:
                num_species += 1
                masses.append(mass)
                symbols.append(symbol[ii])
                

#        sort by mass both masses and symbols
        masses, symbols = zip(*sorted(zip(masses, symbols)))
        for ii in range(len(masses)):
            atm_species += f'{symbols[ii]} '
            mass_lines += f'mass {ii+1} {masses[ii]} #{symbols[ii]}\n'



        input_lammps = f"""####### initialization ############
units metal
atom_style atomic
atom_modify map yes #### QUESTE 2 RIGHE SOLO PER MACE ####
newton on           #### QUESTE 2 RIGHE SOLO PER MACE ####
boundary p p f

timer timeout 23:50:00 every 100
neigh_modify 	every 1 delay 5 check yes

###### atom system definition ###########
read_data  structure.dat
{mass_lines}

###### Interatomic potential ############
pair_style mace no_domain_decomposition
pair_coeff * * potential.dat {atm_species}
variable settemp equal {temp}
velocity  all create {temp} {np.random.randint(1,1000)}

############### time step ############
timestep {dt} 

########## command for simulation #######
# compute press all pressure thermo_temp virial
#compute_modify thermo_press pressure thermo_temp
fix mysim all npt temp {temp} {temp} $(100.0*dt) x {press} {press} $(1000.0*dt) y {press} {press} $(1000.0*dt)
# fix press all 
compute myRDF1 all rdf 1000 1 1
# compute myRDF2 all rdf 1000 1 2
# compute myRDF3 all rdf 1000 2 2
compute myADF all adf 180 1 1 1 1 5 1 5 #&
                        #   1 1 2 1 5 1 5 &
                        #   1 2 2 1 5 1 5 &
                        #   2 2 2 1 5 1 5 &
                        #   2 2 1 1 5 1 5 &
                        #   2 1 1 1 5 1 5
compute mymsd all msd com yes
fix 1 all ave/time 100 1 100 c_myRDF1[*] file rdf11.rdf mode vector
# fix 2 all ave/time 100 1 100 c_myRDF2[*] file rdf12.rdf mode vector
# fix 3 all ave/time 100 1 100 c_myRDF3[*] file rdf22.rdf mode vector
fix 4 all ave/time 1 1 100 c_mymsd[*] file msd.msd
fix 5 all ave/time 100 1 100 c_myADF[*] file adf.adf mode vector
#
thermo		20
thermo_style custom step time temp lx ly lz vol pxx pyy pzz pe ke etotal press spcpu cpuremain
dump mydump all custom 100 coord.atom type id x y z fx fy fz
#dump mydump2 all traj 100 coord.traj
dump mydump3 all atom 100 coord.lammpstrj
#dump out all yaml 100 dump.yaml id type x y z vx vy vz ix iy iz fx fy fz
#dump mydump2 all custom 1000 force.atom type id fx fy fz

############## run simulation ##############
run		{num_steps}
write_data lmp.data
write_restart lmp_restart.rest"""
        
        string_data = io.BytesIO(input_lammps.encode())
        input_file = SinglefileData(file=string_data, filename="input.in")
        return input_file


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
        input_file = self.input_lammps()

        future = self.submit(LammpsCalculation,
                             code=self.inputs.code,
                             structure=self.inputs.structure,
                             potential=self.inputs.potential,
                             input=input_file,
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

