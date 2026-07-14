"""
geometry.py
===========
Convex geometry primitives for Convex Reasoning Networks.

This module implements the geometric building blocks required by the CRN
state-update rule:

    x_{t+1} = Prox_C^M((I + A)x_t + B g_t)

Specifically it provides:

* :class:`ConvexSet`            — abstract base class for convex constraint sets
* :class:`ConvexHullContext`    — convex hull of learnable context vectors C
* :class:`SimplexConstraint`   — probability simplex Δ^{n-1}
* :func:`project_onto_simplex` — Duchi et al. (2008) O(n log n) simplex projection
* :func:`convex_combination`   — weighted combination of a matrix of atoms
* :func:`is_in_convex_hull`    — membership test (for diagnostics)
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ConvexSet(abc.ABC, nn.Module):
    """
    Abstract base class for a closed convex constraint set.

    Sub-classes must implement:

    * :meth:`project` — Euclidean projection onto the set.
    * :meth:`contains` — membership predicate (used in tests / diagnostics).

    All operations are *differentiable* where possible so that gradients can
    flow through projection steps (e.g. via the implicit function theorem or
    analytic gradients of the projection).
    """

    @abc.abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """
        Compute the Euclidean projection of ``x`` onto this set.

        Parameters
        ----------
        x:
            Input tensor of shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Projected tensor with the same shape as ``x``.
        """
        ...

    @abc.abstractmethod
    def contains(self, x: Tensor, tol: float = 1e-6) -> Tensor:
        """
        Test whether each point in ``x`` belongs to the set.

        Parameters
        ----------
        x:
            Input tensor of shape ``(*, dim)``.
        tol:
            Numerical tolerance for membership checks.

        Returns
        -------
        Tensor
            Boolean tensor of shape ``(*,)``.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Alias for :meth:`project` so the module is callable."""
        return self.project(x)


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------


def project_onto_simplex(v: Tensor, z: float = 1.0) -> Tensor:
    """
    Project a batch of vectors onto the probability simplex Δ^{n-1}.

    Implements the O(n log n) algorithm of Duchi et al. (2008):
    *Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions*.

    The projection solves::

        min_{w} ½ ‖w - v‖² subject to w ≥ 0, 1^T w = z.

    Parameters
    ----------
    v:
        Input tensor of shape ``(*, n)``.  Each row is projected
        independently.
    z:
        Simplex radius (default 1.0 for the standard probability simplex).

    Returns
    -------
    Tensor
        Projected tensor of shape ``(*, n)``, each row on the simplex.
    """
    # Flatten to 2-D, project, then reshape
    shape = v.shape
    v2d = v.reshape(-1, shape[-1])           # (batch, n)
    n = v2d.shape[-1]

    # Sort descending
    u, _ = torch.sort(v2d, dim=-1, descending=True)

    # Cumulative sum and thresholding
    cssv = torch.cumsum(u, dim=-1)           # (batch, n)
    rho_range = torch.arange(1, n + 1, dtype=v.dtype, device=v.device)  # (n,)
    # condition: u[j] - (cssv[j] - z) / (j+1) > 0
    mask = u - (cssv - z) / rho_range > 0   # (batch, n)
    # rho = last index where condition holds (1-indexed)
    rho = mask.long().sum(dim=-1) - 1        # (batch,)  0-indexed
    rho = rho.clamp(min=0)

    # θ = (cssv[rho] - z) / (rho + 1)
    theta = (cssv.gather(1, rho.unsqueeze(1)).squeeze(1) - z) / (rho + 1).float()

    w = (v2d - theta.unsqueeze(1)).clamp(min=0.0)
    return w.reshape(shape)


class SimplexConstraint(ConvexSet):
    """
    The probability simplex Δ^{n-1} = {w ∈ ℝ^n : w ≥ 0, 1^T w = 1}.

    Parameters
    ----------
    dim:
        Dimension ``n`` of the simplex.
    z:
        Simplex radius (default 1.0).
    """

    def __init__(self, dim: int, z: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.z = z

    def project(self, x: Tensor) -> Tensor:
        """Project onto the probability simplex via :func:`project_onto_simplex`."""
        return project_onto_simplex(x, z=self.z)

    def contains(self, x: Tensor, tol: float = 1e-6) -> Tensor:
        """Return True for each row that lies on the simplex (within ``tol``)."""
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        nonneg = (x2d >= -tol).all(dim=-1)
        unit_sum = (x2d.sum(dim=-1) - self.z).abs() < tol
        result = nonneg & unit_sum
        return result.reshape(shape[:-1])


# ---------------------------------------------------------------------------
# Convex hull of context vectors
# ---------------------------------------------------------------------------


class ConvexHullContext(ConvexSet):
    """
    Convex hull of a learnable set of context prototype vectors.

    The context set is defined as::

        C = conv({c_1, …, c_K})  ⊂  ℝ^{state_dim}

    where ``c_1, …, c_K`` are learnable parameters.

    A point ``x`` is represented inside ``C`` via convex combination weights
    ``α ∈ Δ^{K-1}`` such that ``x = Σ_k α_k c_k``.

    Parameters
    ----------
    state_dim:
        Dimensionality of the ambient space.
    n_context_vectors:
        Number of prototype vectors ``K``.
    init_scale:
        Standard deviation for random initialisation of context vectors.
    """

    def __init__(
        self,
        state_dim: int,
        n_context_vectors: int,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_context_vectors = n_context_vectors

        # Learnable context prototype matrix — shape: (K, state_dim)
        self.prototypes: nn.Parameter = nn.Parameter(
            torch.empty(n_context_vectors, state_dim)
        )
        self._init_parameters(init_scale)

    def _init_parameters(self, scale: float) -> None:
        """Initialise prototype vectors with a scaled normal distribution."""
        nn.init.normal_(self.prototypes, mean=0.0, std=scale)

    def project(self, x: Tensor) -> Tensor:
        """
        Project ``x`` onto the convex hull via Frank-Wolfe / alternating
        projection (see :mod:`solvers`).

        For the projection step used inside the CRN forward pass, this method
        delegates to the analytic solver when the hull is a polytope defined by
        a small set of vertices (the typical case).

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, state_dim)`` or ``(state_dim,)``.

        Returns
        -------
        Tensor
            Projection of the same shape as ``x``.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Find the best convex combination weights α ∈ Δ^{K-1}
        alpha = self.encode(x)                   # (batch, K)
        projected = self.decode(alpha)            # (batch, state_dim)

        if squeeze:
            projected = projected.squeeze(0)
        return projected

    def contains(self, x: Tensor, tol: float = 1e-6) -> Tensor:
        """
        Approximate membership test via LP feasibility.

        Parameters
        ----------
        x:
            Tensor of shape ``(*, state_dim)``.
        tol:
            Tolerance for the residual.

        Returns
        -------
        Tensor
            Boolean tensor of shape ``(*,)``.
        """
        shape = x.shape[:-1]
        x2d = x.reshape(-1, self.state_dim)
        alpha = self.encode(x2d)                  # (batch, K)
        reconstructed = self.decode(alpha)         # (batch, state_dim)
        residual = (x2d - reconstructed).norm(dim=-1)
        result = residual < tol
        return result.reshape(shape) if shape else result.squeeze()

    def encode(self, x: Tensor) -> Tensor:
        """
        Find convex combination weights α for each point in ``x``.

        Solves the non-negative least-squares problem::

            min_{α ≥ 0, 1^T α = 1} ‖x - C^T α‖²

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, state_dim)``.

        Returns
        -------
        Tensor
            Weight tensor of shape ``(batch, K)``, each row on the simplex.
        """
        # C: (K, state_dim), x: (batch, state_dim)
        # Compute inner products: scores[b, k] = <x[b], c_k>
        # Project the dot-product scores onto simplex as warm start
        C = self.prototypes                       # (K, state_dim)
        # Gram matrix: (K, K)
        G = C @ C.t()
        # Linear term: b[b, k] = x[b] @ c_k — note: b_k = -x^T c_k for min
        # We solve: min_{α ∈ Δ} ½ α^T G α - (x @ C^T) α
        # Use projected gradient descent on the simplex
        batch = x.shape[0]
        K = self.n_context_vectors

        # Warm-start: project correlation scores
        scores = x @ C.t()                        # (batch, K)
        alpha = project_onto_simplex(scores)      # (batch, K)

        # Step size: 1 / λ_max(G)
        eigvals = torch.linalg.eigvalsh(G)
        lam_max = eigvals.max().clamp(min=1e-6)
        step = 1.0 / lam_max.item()

        for _ in range(30):
            # Gradient: G α - (x @ C^T) = G α - scores
            grad = alpha @ G.t() - scores         # (batch, K)
            alpha_new = project_onto_simplex(alpha - step * grad)
            if (alpha_new - alpha).norm() < 1e-7:
                alpha = alpha_new
                break
            alpha = alpha_new

        return alpha

    def decode(self, alpha: Tensor) -> Tensor:
        """
        Reconstruct a state from convex combination weights.

        Parameters
        ----------
        alpha:
            Weight tensor of shape ``(batch, K)`` on the probability simplex.

        Returns
        -------
        Tensor
            Reconstructed state of shape ``(batch, state_dim)``.
        """
        # alpha: (batch, K), prototypes: (K, state_dim)
        return alpha @ self.prototypes             # (batch, state_dim)

    @property
    def diameter(self) -> Tensor:
        """Diameter of the convex hull (max pairwise distance between prototypes)."""
        C = self.prototypes                        # (K, state_dim)
        # Pairwise squared distances
        diffs = C.unsqueeze(0) - C.unsqueeze(1)   # (K, K, state_dim)
        dists = diffs.norm(dim=-1)                 # (K, K)
        return dists.max()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def convex_combination(weights: Tensor, atoms: Tensor) -> Tensor:
    """
    Compute a weighted combination of a matrix of atoms.

    Parameters
    ----------
    weights:
        Tensor of shape ``(batch, K)``; each row should lie on the simplex.
    atoms:
        Tensor of shape ``(K, dim)`` — the atom matrix (e.g. context prototypes).

    Returns
    -------
    Tensor
        Result of shape ``(batch, dim)`` = ``weights @ atoms``.
    """
    return weights @ atoms


def is_in_convex_hull(
    point: Tensor,
    vertices: Tensor,
    tol: float = 1e-6,
) -> bool:
    """
    Check whether ``point`` lies in the convex hull of ``vertices``.

    Uses a linear-programming feasibility check.  Intended for unit tests and
    diagnostic logging, not for use in the training loop.

    Parameters
    ----------
    point:
        1-D tensor of shape ``(dim,)``.
    vertices:
        2-D tensor of shape ``(K, dim)``.
    tol:
        Feasibility tolerance.

    Returns
    -------
    bool
    """
    _, residual = barycentric_coordinates(point, vertices)
    return residual < tol


def barycentric_coordinates(
    point: Tensor,
    vertices: Tensor,
) -> Tuple[Tensor, float]:
    """
    Compute barycentric coordinates of ``point`` w.r.t. ``vertices``.

    Solves the constrained least-squares::

        min_{α} ‖point - vertices^T α‖  s.t. α ≥ 0, 1^T α = 1.

    Parameters
    ----------
    point:
        1-D tensor of shape ``(dim,)``.
    vertices:
        2-D tensor of shape ``(K, dim)``.

    Returns
    -------
    tuple
        ``(alpha, residual)`` where ``alpha`` is the weight vector and
        ``residual`` is the reconstruction error.
    """
    K, dim = vertices.shape
    # Solve via projected gradient descent on the simplex
    G = vertices @ vertices.t()                   # (K, K)
    b = vertices @ point                          # (K,)

    # Warm-start
    scores = vertices @ point                     # (K,)
    alpha = project_onto_simplex(scores.unsqueeze(0)).squeeze(0)  # (K,)

    eigvals = torch.linalg.eigvalsh(G)
    lam_max = eigvals.max().clamp(min=1e-8)
    step = 1.0 / lam_max.item()

    for _ in range(200):
        grad = G @ alpha - b                      # (K,)
        alpha_new = project_onto_simplex((alpha - step * grad).unsqueeze(0)).squeeze(0)
        if (alpha_new - alpha).norm().item() < 1e-9:
            alpha = alpha_new
            break
        alpha = alpha_new

    reconstruction = vertices.t() @ alpha         # (dim,)
    residual = (point - reconstruction).norm().item()
    return alpha, residual

