"""Training workchain for AiiDA TrainsPot."""

import itertools
import random

from aiida.engine import WorkChain, append_, calcfunction
from aiida.orm import Bool, Float, FolderData, Int, Str
from aiida.plugins import DataFactory, WorkflowFactory

MetaWorkChain = WorkflowFactory("trains_pot.metatrain")
MaceWorkChain = WorkflowFactory("trains_pot.macetrain")
PESData = DataFactory("pesdata")

DEFAULT_NON_TRAIN_FRAC = Float(0.2)
DEFAULT_SEED = Int(42)


@calcfunction
def SplitDataset(
    dataset,
    non_training_fraction=DEFAULT_NON_TRAIN_FRAC,
    seed=DEFAULT_SEED,
):
    """Split dataset preserving groups with stochastic rounding."""
    data = dataset.get_list()

    if not (0.0 < non_training_fraction.value < 1.0):
        raise ValueError("non_training_fraction must be between 0 and 1")

    train_p = 1.0 - non_training_fraction.value

    rng = random.Random(seed.value)

    exclude_list = [
        "energy",
        "cell",
        "stress",
        "forces",
        "symbols",
        "positions",
        "id_lammps",
        "input_structure_uuid",
        "sigma_strain",
    ]

    def check_exclude(key):
        return not any(el in key for el in exclude_list)

    def get_grouping_key(d):
        return tuple((k, v) for k, v in d.items() if check_exclude(k))

    sorted_data = sorted(data, key=get_grouping_key)
    grouped_data = itertools.groupby(sorted_data, key=get_grouping_key)

    training_set = []
    validation_set = []
    test_set = []

    for _, group in grouped_data:
        group_list = list(group)

        # ---- Forced assignment ----
        if "gen_method" in group_list[0]:
            if (
                group_list[0]["gen_method"] in ["INPUT_STRUCTURE", "ISOLATED_ATOM", "EQUILIBRIUM"]
                or len(group_list[0]["positions"]) == 1
            ):
                training_set += group_list
                continue
        if "set" in group_list[0]:
            label = group_list[0]["set"]
            if label == "TRAINING":
                training_set += group_list
                continue
            if label == "VALIDATION":
                validation_set += group_list
                continue
            if label == "TEST":
                test_set += group_list
                continue

        rng.shuffle(group_list)

        total = len(group_list)
        exp_train = int(total * train_p)
        # Split the data into sets
        training_set += group_list[:exp_train]
        validation_set += group_list[exp_train:][::2]
        test_set += group_list[exp_train:][1::2]

    # ---- tagging ----
    def tag(lst, label):
        for el in lst:
            el["set"] = label
            if "gen_method" not in el:
                el["gen_method"] = "UNKNOWN"

    tag(training_set, "TRAINING")
    tag(validation_set, "VALIDATION")
    tag(test_set, "TEST")

    pes_training_set = PESData()
    pes_training_set.set_list(training_set)

    pes_validation_set = PESData()
    pes_validation_set.set_list(validation_set)

    pes_test_set = PESData()
    pes_test_set.set_list(test_set)

    pes_global = PESData()
    pes_global.set_list(training_set + validation_set + test_set)

    return {
        "train_set": pes_training_set,
        "validation_set": pes_validation_set,
        "test_set": pes_test_set,
        "global_splitted": pes_global,
    }


class TrainingWorkChain(WorkChain):
    """A workchain to loop over structures and submit MACEWorkChain."""

    ######################################################
    ##                 DEFAULT VALUES                   ##
    ######################################################
    DEFAULT_training_engine = "MACE"

    ACCEPTED_ENGINES = ["MACE", "META"]
    ######################################################

    @classmethod
    def define(cls, spec):
        """Input and output specification."""
        super().define(spec)
        spec.input("num_potentials", valid_type=Int, default=lambda: Int(1), required=False)
        spec.input("non_training_fraction", valid_type=Float, default=lambda: Float(0.2), required=False)
        spec.input(
            "engine",
            default=lambda: Str(cls.DEFAULT_training_engine),
            valid_type=Str,
            help="Training engine",
            required=True,
            validator=cls.validate_engine,
        )
        spec.input(
            "dataset",
            valid_type=PESData,
            help="Training dataset",
        )
        spec.input_namespace(
            "checkpoints",
            valid_type=FolderData,
            required=False,
            help="Checkpoints file",
        )
        spec.expose_inputs(
            MaceWorkChain,
            namespace="mace",
            exclude=("train.training_set", "train.validation_set", "train.test_set"),
            namespace_options={"validator": None, "required": False, "populate_defaults": False},
        )
        spec.expose_inputs(
            MetaWorkChain,
            namespace="meta",
            exclude=("train.training_set", "train.validation_set", "train.test_set"),
            namespace_options={"validator": None, "required": False, "populate_defaults": False},
        )

        spec.inputs.validator = cls.validate_inputs
        spec.output_namespace("training", dynamic=True, help="Training outputs")
        spec.output(
            "global_splitted",
            valid_type=PESData,
        )
        spec.outline(cls.run_training, cls.finalize)

    @classmethod
    def validate_inputs(cls, inputs, _):
        """Validate the inputs based on the selected engine."""
        engine = inputs["engine"].value
        if engine == "MACE" and "mace" not in inputs:
            return "Missing required `mace` inputs for engine='MACE'."
        if engine == "META" and "meta" not in inputs:
            return "Missing required `meta` inputs for engine='META'."
        return None

    @classmethod
    def validate_engine(cls, value, _):
        """Validate the engine input."""
        if value.value not in cls.ACCEPTED_ENGINES:
            return f"The `engine` input can only be {', '.join(cls.ACCEPTED_ENGINES)}."

    def run_training(self):
        """Run training."""
        split_datasets = SplitDataset(self.inputs.dataset, self.inputs.non_training_fraction)
        train_set = split_datasets["train_set"]
        validation_set = split_datasets["validation_set"]
        test_set = split_datasets["test_set"]

        self.out("global_splitted", split_datasets["global_splitted"])

        self.report(f"Training set size: {len(train_set.get_list())}")
        self.report(f"Validation set size: {len(validation_set.get_list())}")
        self.report(f"Test set size: {len(test_set.get_list())}")

        if self.inputs.engine.value == "MACE":
            inputs = self.exposed_inputs(MaceWorkChain, namespace="mace")

            inputs.train["training_set"] = train_set
            inputs.train["validation_set"] = validation_set
            inputs.train["test_set"] = test_set

            if "checkpoints" in self.inputs:
                inputs["checkpoints"] = self.inputs.checkpoints
                inputs.train["restart"] = Bool(True)

            if "checkpoints" in inputs:
                chkpts = list(dict(inputs.checkpoints).values())

            for ii in range(self.inputs.num_potentials.value):
                if "checkpoints" in self.inputs and ii < len(chkpts):
                    inputs.train["checkpoints"] = chkpts[ii]

                inputs.train["index_pot"] = Int(ii)
                future = self.submit(MaceWorkChain, **inputs)
                self.to_context(mace_wc=append_(future))
            pass

        if self.inputs.engine.value == "META":
            inputs = self.exposed_inputs(MetaWorkChain, namespace="meta")

            inputs.train["training_set"] = train_set
            inputs.train["validation_set"] = validation_set
            inputs.train["test_set"] = test_set

            if "checkpoints" in self.inputs:
                inputs["checkpoints"] = self.inputs.checkpoints
                inputs.train["restart"] = Bool(True)

            if "checkpoints" in inputs:
                chkpts = list(dict(inputs.checkpoints).values())

            for ii in range(self.inputs.num_potentials.value):
                if "checkpoints" in self.inputs and ii < len(chkpts):
                    inputs.train["checkpoints"] = chkpts[ii]

                inputs.train["index_pot"] = Int(ii)
                future = self.submit(MetaWorkChain, **inputs)
                self.to_context(meta_wc=append_(future))
            pass

    def finalize(self):
        """Collect and output results from all WorkChain instances."""
        results = {}

        if self.inputs.engine.value == "MACE":
            for ii, calc in enumerate(self.ctx.mace_wc):
                results[f"mace_{ii}"] = {}
                for el in calc.outputs:
                    results[f"mace_{ii}"][el] = calc.outputs[el]

        if self.inputs.engine.value == "META":
            for ii, calc in enumerate(self.ctx.meta_wc):
                results[f"meta_{ii}"] = {}
                for el in calc.outputs:
                    results[f"meta_{ii}"][el] = calc.outputs[el]

        self.out("training", results)
