from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json

@dataclass
class CustomArguments:
    r"""
    Arguments pertaining to the custom usage.
    """
    training_stage: int = field(
        default=1,
        metadata={"help": "Training stage of our Causal Analyst."},
    )
    num_classes: int = field(
        default=5,
        metadata={"help": "Number of classes for classification."},
    )
    num_feature_nodes: int = field(
        default=38,
        metadata={"help": "Number of feature nodes for classification."},
    )
    feature_subset: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Subset of features to use."},
    )
    feature_name_path: Optional[str] = field(
        default="/data/home/Licheng/workspace/LLaMA-Factory/prior_knowledge/feature_name_mapping.json",
        metadata={"help": "Path to the feature name file."},
    )
    feature_name_in: Optional[List[str]] = field(
        default=None,
    )
    feature_name_out: Optional[List[str]] = field(
        default=None,
    )
    feature_names: Optional[List[str]] = field(
        default=None,
    )
    forbidden_edges_filepath: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the forbidden edges file."},
    )
    positive_required_edges_filepath: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the positive required edges file."},
    )
    negative_required_edges_filepath: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the negative required edges file."},
    )
    alignment_type: str = field(
        default="additive",
        metadata={"help": "Type of alignment to use."},
    )
    x_dim: int = field(
        default=1,
        metadata={"help": "Dimensions of the input."},
    )
    z_dim: int = field(
        default=1,
        metadata={"help": "Dimension of the latent space."},
    )
    graph_threshold: float = field(
        default=0.3,
        metadata={"help": "Threshold for graph adjacency."},
    )
    tau_A: float = field(
        default=0.0,
        metadata={"help": "Parameter for the A matrix."},
    )
    lambda_A: float = field(
        default=0.0,
        metadata={"help": "Parameter for the A matrix."},
    )
    c_A: float = field(
        default=1.0,
        metadata={"help": "Parameter for the A matrix."},
    )
    use_A_connect_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use A connect loss."},
    )
    use_A_positiver_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use A positive loss."},
    )
    encoder_hidden_size: int = field(
        default=128,
        metadata={"help": "Hidden size of the encoder."},
    )
    decoder_hidden_size: int = field(
        default=128,
        metadata={"help": "Hidden size of the decoder."},
    )
    h_tol: float = field(
        default=1e-8,
        metadata={"help": "Tolerance for h."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Parameter for the model."},
    )
    prior: bool = field(
        default=False,
        metadata={"help": "Whether to use prior knowledge."},
    )
    extra_modules_to_train: Optional[str] = field(
        default=None,
        metadata={"help": "Extra modules to train."},
    )
    recursive_dag_search: bool = field(
        default=False,
        metadata={"help": "Whether to use recursive DAG search."},
    )
    first_train: bool = field(
        default=False,
        metadata={"help": "Whether this is the first training."},
    )
    cls_lambda: float = field(
        default=1.0,
        metadata={"help": "Parameter for the classification loss."},
    )
    align_lambda: float = field(
        default=1.0,
        metadata={"help": "Parameter for the alignment loss."},
    )
    elbo_lambda: float = field(
        default=1.0,
        metadata={"help": "Parameter for the ELBO loss."},
    )
    rec_lambda: float = field(
        default=1.0,
        metadata={"help": "Parameter for the reconstruction loss."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.extra_modules_to_train = split_arg(self.extra_modules_to_train)

        try:
            with open(self.feature_name_path, "r") as f:
                feature_mapping = json.load(f)
            feature_names = list(feature_mapping.keys())
            self.feature_name_in = feature_names[:-self.num_classes]
            self.feature_name_out = feature_names[-self.num_classes:]
        except Exception as e:
            raise ValueError(f"Failed to load feature names from {self.feature_name_path}: {e}")

        if self.feature_subset is not None:
            self.num_feature_nodes = len(self.feature_subset)
            self.feature_name_in = [self.feature_name_in[i] for i in self.feature_subset]

        self.num_feature_nodes += self.num_classes
        self.feature_names = self.feature_name_in + self.feature_name_out

        assert self.alignment_type in ["additive", "multiplicative", 'attentive'], f"Invalid alignment type: {self.alignment_type}"

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
