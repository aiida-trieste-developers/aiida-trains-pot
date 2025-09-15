Workchains
==========

.. toctree::
   :maxdepth: 3

   trains_pot_wc
   augmentation_wc
   abinitio_labelling_wc
   training_wc
   exploration_wc
   mace_wc

.. currentmodule:: aiida_trains_pot


Below are the primary workchains available in the module:



**Main TrainsPot WorkChain**
  .. autoclass:: aiida_trains_pot.aiida_trains_pot_workflow.aiida_trains_pot_workflow.TrainsPotWorkChain
     :members:
     :exclude-members: DEFAULT_RATTLE_rattle_fraction     

**Dataset Augmentation WorkChain**
  .. autoclass:: aiida_trains_pot.datasetaugmentation.datasetaugmentation_wc.datasetaugmentation_wc.DatasetAugmentationWorkChain
     :members:
  
**Ab Initio Labelling WorkChain**
  .. autoclass:: aiida_trains_pot.aiida_trains_pot_workflow.abinitiolabelling_wc.AbInitioLabellingWorkChain
     :members:
    
**Training WorkChain**
  .. autoclass:: aiida_trains_pot.aiida_trains_pot_workflow.training_wc.TrainingWorkChain
     :members:
  
**MD Exploration WorkChain**
  .. autoclass:: aiida_trains_pot.aiida_trains_pot_workflow.exploration_wc.ExplorationWorkChain
     :members:

**MACE Training WorkChain**
  .. autoclass:: aiida_trains_pot.mace.mace_train_wc.mace_train_wc.MaceTrainWorkChain
     :members:



