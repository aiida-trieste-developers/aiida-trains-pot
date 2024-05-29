# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, Float, SinglefileData, List
from aiida.plugins import CalculationFactory
from ase.io.lammpsrun import read_lammps_dump_text
from io import StringIO
import numpy as np
import os
import io
load_profile()

# LammpsCalculation = CalculationFactory('lammps_base')


@calcfunction
def extract_frames(input_structure, dt, correlation_time, saving_frequency, **trajectories):
    """Extract frames from trajectory."""

    input_structure = input_structure.get_ase()
    masses = []
    symbols = []
    symbol = input_structure.get_chemical_symbols()
    for ii, mass in enumerate(input_structure.get_masses()):
        if mass not in masses:
            masses.append(mass)
            symbols.append(symbol[ii])
        
    masses, symbols = zip(*sorted(zip(masses, symbols)))
    # print(list(symbols))
    extracted_frames = []
    for _, trajectory in trajectories.items():
        trajectory_frames = read_lammps_dump_text(StringIO(trajectory.get_content()), index=slice(0, int(1e50), 1), specorder=list(symbols))
        
        i = 0
        while i < len(trajectory_frames):
            extracted_frames.append({'cell': List(list(trajectory_frames[i].get_cell())),
                    'symbols': List(list(trajectory_frames[i].get_chemical_symbols())),
                    'positions': List(list(trajectory_frames[i].get_positions())),
                    'md_forces': List(list(trajectory_frames[i].get_forces())),
                    'gen_method': Str('LAMMPS')
                    })
            i = i + int(correlation_time/dt/saving_frequency)

    return {'extracted_frames_list': List(list=extracted_frames)}


class LammpsFrameExtraction(WorkChain):
    """WorkChain to launch LAMMPS calculations."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        # spec.expose_inputs(LammpsCalculation, namespace="lmp", exclude=('input','structure','potential'), namespace_options={'validator': None})
        # spec.expose_outputs(LammpsCalculation, namespace="lmp_out")
        # # spec.expose_inputs(LammpsCalculation, namespace="lmp")

        spec.input_namespace("trajectories", valid_type=SinglefileData, help="Trajectory to extract frames from.")
        spec.input("input_structure", valid_type=StructureData, help="Input structure file for LAMMPS.")
        spec.input("dt", valid_type=Float, help="Time step.")
        spec.input("correlation_time", valid_type=Float, help="Correlation time after which to extract frames.")
        spec.input("saving_frequency", valid_type=Int, help="Frequency of saving frames.")

        spec.output("extracted_frames_list", valid_type=List, help="List of extracted frames.")

        spec.outline(
            cls.run_extraction,
            # cls.finalize,
            # cls.save_files
        )
        

    def run_extraction(self):
        """Run Lammps calculations."""

        self.report('type', type(self.inputs.trajectories))
        for key, value in self.inputs.trajectories.items():
            self.report("key", {key})
        
        self.out("extracted_frames_list", extract_frames(self.inputs.input_structure, self.inputs.dt, self.inputs.correlation_time, self.inputs.saving_frequency)["extracted_frames_list"], **self.inputs.trajectories)
            

    


    # def finalize(self):
    #     """Finalize."""


    #     self.out_many(self.exposed_outputs(self.ctx.lammps_calculations[0], LammpsCalculation, namespace="lmp_out"))
    #     # for ii, val in enumerate(self.ctx.lammps_calculations):
    #     #     self.out(f'rdf', val.outputs.rdf)
    #     #     self.out(f'coord_lmmpstrj', val.outputs.coord_lmmpstrj)
    #     #     self.out(f'msd', val.outputs.msd)
    #     #     self.out(f'lammps_out', val.outputs.lammps_out)
    #     #     self.out(f'final_structure', val.outputs.final_structure)
    #     #     self.out(f'lmp_restart', val.outputs.lmp_restart)
    #     #     self.out(f'coord_atom', val.outputs.coord_atom)
    #     #     self.out(f'log_lammps', val.outputs.log_lammps)



    # def save_files(self):
    #     """Create folder and save files."""
    #     folder = f'{self.inputs.parent_folder.value}/Data/LAMMPS_pot{self.inputs.potential.pk}_str{self.inputs.structure.pk}_DT{self.inputs.dt.value}_N{self.inputs.num_steps.value}_P{self.inputs.pressure.value}_T{self.inputs.temperature.value}'
    #     os.makedirs(folder, exist_ok=True)
    #     retrived = self.ctx.lammps_calculations[0].get_retrieved_node()


    #     for r in retrived.list_objects():
    #         self.report(r.name)
    #         with retrived.open(r.name, 'rb') as handle:
    #             with open(f'{folder}/{r.name}', "w") as f:
    #                 try:
    #                     f.write(handle.read().decode('utf-8'))
    #                 except:
    #                     pass

