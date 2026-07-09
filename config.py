"""
config.py
=========
Centralised experiment configuration for Convex Reasoning Networks.

All hyper-parameters, paths, and reproducibility settings live here.
A single :class:`CRNConfig` dataclass is the authoritative source of truth
consumed by every other module.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).parent.resolve()
CHECKPOINTS_DIR: Path = ROOT_DIR / "checkpoints"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = ROOT_DIR / "figures"
PAPER_DIR: Path = ROOT_DIR / "paper"

for _d in (CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Solver literal type
# ---------------------------------------------------------------------------

SolverName = Literal["analytic", "pgd", "frank_wolfe"]
MetricName = Literal["spd", "euclidean"]


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Parameters for synthetic trajectory generation."""

    state_dim: int = 8
    """Dimensionality of the state vector x_t."""

    input_dim: int = 4
    """Dimensionality of the exogenous input g_t."""

    context_weight_dim: int = 16
    """
    Dimensionality of the convex-combination weight vector produced by the
    dataset, i.e. the number of context prototype atoms K.  Must match
    ``ModelConfig.n_context_vectors``.
    """

    n_trajectories: int = 2_000
    """Total number of trajectories to generate."""

    trajectory_length: int = 32
    """Number of time steps T per trajectory."""

    noise_std: float = 0.02
    """Standard deviation of additive Gaussian observation noise."""

    train_frac: float = 0.70
    """Fraction of data used for training."""

    val_frac: float = 0.15
    """Fraction of data used for validation (remainder → test)."""

    seed: int = 0
    """Master seed for dataset generation."""


@dataclass
class ModelConfig:
    """Architecture parameters for the CRN model."""

    state_dim: int = 8
    """Must match DataConfig.state_dim."""

    input_dim: int = 4
    """Must match DataConfig.input_dim."""

    n_context_vectors: int = 16
    """
    Number of learnable context prototype vectors K (the atoms of the convex
    hull C = conv({c_1, …, c_K})).  Must match DataConfig.context_weight_dim
    so that dataset-generated convex-combination weights have the correct size.
    """

    metric_eps: float = 1e-4
    """Regularisation ε added to LL^T to guarantee positive-definiteness."""

    contraction_factor: float = 0.9
    """Upper bound on the spectral norm of A (contractivity constraint)."""

    solver: SolverName = "analytic"
    """Which prox solver to use during forward passes."""

    metric_type: MetricName = "spd"
    """Whether to use the full learnable SPD metric or a fixed Euclidean one."""


@dataclass
class SolverConfig:
    """Hyper-parameters shared across iterative prox solvers (PGD, Frank-Wolfe)."""

    max_iter: int = 50
    """Maximum number of solver iterations."""

    tol: float = 1e-6
    """Convergence tolerance (norm of successive iterates)."""

    pgd_step_size: float = 0.1
    """Step size η for the Projected Gradient Descent solver."""

    fw_line_search: bool = True
    """Whether Frank-Wolfe uses exact line search (True) or fixed step (False)."""


@dataclass
class TrainConfig:
    """Training loop parameters."""

    epochs: int = 200
    """Maximum number of training epochs."""

    batch_size: int = 64
    """Mini-batch size."""

    learning_rate: float = 3e-4
    """Initial Adam learning rate."""

    weight_decay: float = 1e-5
    """L2 regularisation coefficient."""

    lr_scheduler: Literal["cosine", "step", "none"] = "cosine"
    """Which learning-rate scheduler to use."""

    lr_step_size: int = 50
    """Step size for StepLR (ignored if scheduler != 'step')."""

    lr_gamma: float = 0.5
    """Decay factor for StepLR (ignored if scheduler != 'step')."""

    early_stopping_patience: int = 20
    """Number of validation epochs without improvement before stopping."""

    gradient_clip_norm: float = 1.0
    """Maximum gradient norm for gradient clipping (0 disables clipping)."""

    checkpoint_every: int = 10
    """Save a checkpoint every N epochs (in addition to best-model saves)."""

    seed: int = 42
    """Master seed for training (weight init, dataloader shuffles)."""

    device: str = "cpu"
    """Torch device string, e.g. 'cpu', 'cuda', 'cuda:0', 'mps'."""

    num_workers: int = 0
    """DataLoader worker processes (0 = main process only)."""


@dataclass
class EvalConfig:
    """Parameters used during evaluation and ablation runs."""

    n_rollout_steps: int = 64
    """Number of auto-regressive steps for convergence-rate evaluation."""

    timing_repeats: int = 10
    """Number of timed forward passes to average for execution-time estimates."""

    memory_profiling: bool = True
    """Whether to record peak GPU/CPU memory during evaluation."""


@dataclass
class AblationConfig:
    """Specification of the ablation study grid."""

    run_metric_ablation: bool = True
    """Compare SPD metric vs. Euclidean metric."""

    run_solver_ablation: bool = True
    """Compare analytic / PGD / Frank-Wolfe solvers."""

    n_seeds: int = 3
    """Number of independent seeds per ablation cell."""


@dataclass
class CRNConfig:
    """
    Top-level experiment configuration.

    A single instance of this class is passed through the entire pipeline so
    that every module reads from one authoritative source.

    Example
    -------
    >>> cfg = CRNConfig()
    >>> cfg.train.learning_rate
    0.0003
    """

    experiment_name: str = "crn_baseline"
    """Human-readable experiment identifier (used in filenames)."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # ------------------------------------------------------------------
    # Derived / metadata fields (populated automatically)
    # ------------------------------------------------------------------

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    platform_info: str = field(default_factory=platform.platform)

    def __post_init__(self) -> None:
        """Validate cross-field consistency and propagate shared dimensions."""
        # Propagate dimension consistency between data and model configs.
        # The single canonical K is data.context_weight_dim / model.n_context_vectors.
        self.model.state_dim = self.data.state_dim
        self.model.input_dim = self.data.input_dim
        # Keep n_context_vectors in sync with the dataset's weight vector size.
        self.data.context_weight_dim = self.model.n_context_vectors

        if not (0.0 < self.data.train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1).")
        if not (0.0 < self.data.val_frac < 1.0):
            raise ValueError("val_frac must be in (0, 1).")
        if self.data.train_frac + self.data.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1 to leave room for test set.")

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dictionary representation (JSON-serialisable)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Persist the configuration to disk as JSON.

        Parameters
        ----------
        path:
            Target file path.  Defaults to
            ``results/<experiment_name>_config.json``.

        Returns
        -------
        Path
            Absolute path of the saved file.
        """
        if path is None:
            path = RESULTS_DIR / f"{self.experiment_name}_config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> "CRNConfig":
        """
        Load a configuration from a JSON file previously saved by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the JSON file.

        Returns
        -------
        CRNConfig
        """
        raw = json.loads(Path(path).read_text())
        data = DataConfig(**raw.pop("data"))
        model = ModelConfig(**raw.pop("model"))
        solver = SolverConfig(**raw.pop("solver"))
        train = TrainConfig(**raw.pop("train"))
        eval_ = EvalConfig(**raw.pop("eval"))
        ablation = AblationConfig(**raw.pop("ablation"))
        # Drop auto-populated metadata fields that the constructor re-generates
        for meta_key in ("timestamp", "hostname", "python_version", "platform_info"):
            raw.pop(meta_key, None)
        return cls(
            data=data,
            model=model,
            solver=solver,
            train=train,
            eval=eval_,
            ablation=ablation,
            **raw,
        )

    def fingerprint(self) -> str:
        """Return a short SHA-256 hex digest of the serialised configuration."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------

    def checkpoint_path(self, tag: str = "best") -> Path:
        """Return the canonical path for a model checkpoint."""
        return CHECKPOINTS_DIR / self.experiment_name / f"{tag}.pt"

    def results_path(self, filename: str) -> Path:
        """Return the canonical path for a results file."""
        return RESULTS_DIR / self.experiment_name / filename

    def figures_path(self, filename: str) -> Path:
        """Return the canonical path for a figure file."""
        return FIGURES_DIR / self.experiment_name / filename

