.. _get-started:

==================================
Get Started with aiida-trains-pot
==================================

Welcome to `aiida-trains-pot <https://github.com/aiida-trieste-developers/aiida-trains-pot>`_! This guide will walk you through setting up and running the graphene example located in `examples/graphene <https://github.com/aiida-trieste-developers/aiida-trains-pot/tree/main/examples/graphene>`_ on GitHub.

Prerequisites
-------------

To use `aiida-trains-pot <https://github.com/aiida-trieste-developers/aiida-trains-pot>`_, ensure you have the following prerequisites installed:

- `AiiDA <https://aiida.net>`_
- `QuantumESPRESSO <https://www.quantum-espresso.org>`_, `MACE <https://github.com/ACEsuit/mace>`_, and `LAMMPS <https://mace-docs.readthedocs.io/en/latest/guide/lammps.html>`_ codes installed on your computing environment and configured in AiiDA
- Access to a GPU-enabled HPC cluster with SLURM support (e.g., Cineca Leonardo)

**Note**: The example in this guide uses QuantumESPRESSO, MACE, and LAMMPS workflows configured with the appropriate GPU parameters.

Step 1: Setup the Environment
-----------------------------

Before starting, ensure you load your AiiDA profile and import necessary dependencies. In this example, the graphene structure `gr8x8.xyz <https://github.com/aiida-trieste-developers/aiida-trains-pot/blob/main/examples/graphene/gr8x8.xyz>`_ and required configuration files are located in the `examples/graphene <https://github.com/aiida-trieste-developers/aiida-trains-pot/tree/main/examples/graphene>`_ directory.

.. code-block:: python

    from aiida.orm import load_code, load_node, load_group, load_computer, Str, Dict, List, Int, Bool, Float, StructureData
    from aiida import load_profile
    from aiida.engine import submit
    from aiida.plugins import WorkflowFactory, DataFactory
    from pathlib import Path
    from aiida.common.extendeddicts import AttributeDict
    from ase.io import read
    import yaml
    import os
    from aiida_trains_pot.utils.restart import models_from_trainingwc
    from aiida_trains_pot.utils.generate_config import generate_lammps_md_config

    load_profile()

Step 2: Define and Load the Codes
---------------------------------

In this example, we use QuantumESPRESSO (QE), MACE, LAMMPS, and committee evaluation codes. Make sure these are installed and available as AiiDA codes. Examples of configuration YAML files can be found in `examples/setup_codes <https://github.com/aiida-trieste-developers/aiida-trains-pot/tree/main/examples/setup_codes>`_:

.. code-block:: python

    QE_code                 = load_code('qe7.2-pw@leo1_scratch_bind')
    MACE_train_code         = load_code('mace_train@leo1_scratch_mace')
    MACE_preprocess_code    = load_code('mace_preprocess@leo1_scratch_mace')
    MACE_postprocess_code   = load_code('mace_postprocess@leo1_scratch_mace')
    LAMMPS_code             = load_code('lmp4mace@leo1_scratch')
    EVALUATION_code         = load_code('committee_evaluation_portable')

Step 3: Set Machine Parameters
------------------------------

Customize machine parameters for each code (time, nodes, GPUs, memory, etc.). Here’s an example for configuring QuantumESPRESSO:

.. code-block:: python

    QE_machine = {
        'time': "00:05:00",
        'nodes': 1,
        'gpu': "1",
        'taskpn': 1,
        'cpupt': "8",
        'mem': "70GB",
        'account': "***",
        'partition': "boost_usr_prod",
        'qos': "boost_qos_dbg"
    }

Repeat this process for each code (MACE, LAMMPS and committee evaluation), adapting the parameters as needed.

Step 4: Load the Graphene Structure
-----------------------------------

Load the graphene structure `gr8x8.xyz <https://github.com/aiida-trieste-developers/aiida-trains-pot/blob/main/examples/graphene/gr8x8.xyz>`_:

.. code-block:: python

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_structures = [StructureData(ase=read(os.path.join(script_dir, 'gr8x8.xyz')))]

Step 5: Setup the TrainsPot Workflow
------------------------------------

The `TrainsPot` workflow combines several tasks. Use `get_builder()` to set up the workflow:

.. code-block:: python

    TrainsPot = WorkflowFactory('trains_pot.workflow')
    builder = TrainsPot.get_builder()
    builder.structures =  {f'structure_{i}':input_structures[i] for i in range(len(input_structures))}
    builder.do_dataset_augmentation = Bool(True)
    builder.do_ab_initio_labelling = Bool(True)
    builder.do_training = Bool(True)
    builder.do_exploration = Bool(True)
    builder.max_loops = Int(1)

Step 6: Configure Dataset Augmentation
--------------------------------------

Set up parameters for data augmentation. You can adjust options like `do_rattle`, `do_input`, and `do_isolated` for custom configurations:

.. code-block:: python

    builder.dataset_augmentation.do_rattle = Bool(True)
    builder.dataset_augmentation.do_input = Bool(True)
    builder.dataset_augmentation.do_isolated = Bool(True)
    builder.dataset_augmentation.rattle.params.rattle_fraction = Float(0.1)
    builder.dataset_augmentation.rattle.params.max_sigma_strain = Float(0.1)
    builder.dataset_augmentation.rattle.params.n_configs = Int(20)
    builder.dataset_augmentation.rattle.params.frac_vacancies = Float(0.1)
    builder.dataset_augmentation.rattle.params.vacancies_per_config = Int(1)

Step 7: Configure Ab Initio Labelling (QuantumESPRESSO)
-------------------------------------------------------

Load QE settings, k-points, and pseudopotentials for labelling:

.. code-block:: python

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])
    pseudo_family = load_group('SSSP/1.3/PBE/precision')
    cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=input_structures[0], unit='Ry')

    builder.ab_initio_labelling.quantumespresso.pw.code = QE_code
    builder.ab_initio_labelling.quantumespresso.pw.pseudos = pseudo_family.get_pseudos(structure=input_structures[0])
    builder.ab_initio_labelling.quantumespresso.kpoints = kpoints

Step 8: Configure MACE and LAMMPS for Training and Exploration
--------------------------------------------------------------

Use a YAML configuration for MACE. The additonal information about the MACE parameters can be found in the `MACE documentation <https://mace-docs.readthedocs.io/en/latest/guide/training.html>`_:

.. code-block:: python

    MACE_config = os.path.join(script_dir, 'mace_config.yml')
    with open(MACE_config, 'r') as yaml_file:
        mace_config = yaml.safe_load(yaml_file)

    builder.training.mace.train.mace_config = Dict(mace_config)
    builder.training.num_potentials = Int(4)

For LAMMPS, load additional parameters from `lammps_md_params.yml <https://github.com/aiida-trieste-developers/aiida-trains-pot/blob/main/examples/graphene/lammps_md_params.yml>`_. The additonal information about the LAMMPS parameters can be found in the `LAMMPS documentation <https://aiida-lammps.readthedocs.io/en/latest/topics/data/parameters.html>`_:

.. code-block:: python

    lammps_params_yaml = os.path.join(script_dir, 'lammps_md_params.yml')
    with open(lammps_params_yaml, 'r') as yaml_file:
        lammps_params_list = yaml.safe_load(yaml_file)
    builder.exploration.params_list = List(lammps_params_list)

Step 9: Setup Committee Evaluation
----------------------------------

Define resources and SLURM options for the committee evaluation stage:

.. code-block:: python

    builder.committee_evaluation.code = EVALUATION_code    

Step 10: Submit the Workflow
----------------------------

Once everything is set up, submit the workflow:

.. code-block:: python

    calc = submit(builder)
    print(f"Submitted calculation with PK = {calc.pk}")

This will submit the workflow to AiiDA, where you can track its progress and analyze results.

---

This guide should help you get started with `aiida-trains-pot <https://github.com/aiida-trieste-developers/aiida-trains-pot>`_! For more information on AiiDA workflows, check the AiiDA documentation.


