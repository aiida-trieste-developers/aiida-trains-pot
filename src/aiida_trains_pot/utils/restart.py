"""Utilities to restart training from previous workchains."""

from aiida.orm import load_node


def models_from_trainingwc(builder, identifier, get_labelled_dataset=False, get_config=False):
    """Return a builder populated with potentials and checkpoints from previous training.

    Args:
        builder: Builder to be populated.
        identifier: Identifier of the training workchain (pk or uuid).
        get_labelled_dataset: If True, the labelled dataset will be added to the builder.
        get_config: If True, the mace configuration parameters will be added to the builder.

    Returns:
        builder: Builder populated with potentials, checkpoints and eventually labelled
        dataset and mace configuration parameters.
    """
    outputs = load_node(identifier).outputs
    models_ase = {}
    models_lammps = {}
    models_checkpoints = {}
    if get_labelled_dataset:
        builder.dataset = outputs["global_splitted"]
    if get_config:
        inputs = load_node(identifier).inputs
        builder.training.mace.train.mace_config = inputs['mace']['train']['mace_config']
    for trainings in outputs['training']:
        if trainings.startswith('mace'):
            if "model_stage2_ase" in outputs['training'][trainings]:
                models_ase[trainings] = outputs['training'][trainings]["model_stage2_ase"]
            elif "model_stage1_ase" in outputs['training'][trainings]:
                models_ase[trainings] = outputs['training'][trainings]["model_stage1_ase"]
            if "model_stage2_lammps" in outputs['training'][trainings]:
                models_lammps[trainings] = outputs['training'][trainings]["model_stage2_lammps"]
            elif "model_stage1_lammps" in outputs['training'][trainings]:
                models_lammps[trainings] = outputs['training'][trainings]["model_stage1_lammps"]
            if "checkpoints" in outputs['training'][trainings]:
                models_checkpoints[trainings] = outputs['training'][trainings]["checkpoints"]
        if trainings.startswith('meta'):            
            if "model_stage2_lammps" in outputs['training'][trainings]:
                models_lammps[trainings] = outputs['training'][trainings]["model_stage2_lammps"]            
            if "checkpoints" in outputs['training'][trainings]:
                models_checkpoints[trainings] = outputs['training'][trainings]["checkpoints"]

    builder.models_ase = models_ase
    builder.models_lammps = models_lammps
    builder.training.checkpoints = models_checkpoints

    return builder


def models_from_aiidatrainspotwc(builder, identifier):
    """Return a builder populated with potentials and checkpoints from previous training.

    Args:
        builder: Builder to be populated.
        identifier: Identifier of the training workchain (pk or uuid).

    Returns:
        builder: Builder populated with dataset, models_ase, models_lammps configuration parameters.
    """
    outputs = load_node(identifier).outputs
    models_ase = {}
    models_lammps = {}
    checkpoints = {}

    for model_ase in outputs["models_ase"]:
        models_ase[model_ase] = outputs["models_ase"][model_ase]
    for model_lammps in outputs["models_lammps"]:
        models_lammps[model_lammps] = outputs["models_lammps"][model_lammps]
    for checkpoint in outputs["checkpoints"]:
        checkpoints[checkpoint] = outputs["checkpoints"][checkpoint]

    builder.dataset = outputs["dataset"]
    builder.models_ase = models_ase
    builder.models_lammps = models_lammps
    builder.training.checkpoints = checkpoints

    return builder
