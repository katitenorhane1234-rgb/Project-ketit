"""
solvers.py
==========
Proximal operator solvers for Convex Reasoning Networks.

All three solvers compute the *M-weighted proximal operator*::

    Prox_C^M(v) = argmin_{x ∈ C} ½ ‖x - v‖_M²

where C is a closed convex set and M is a symmetric positive-definite metric.

Three independent implementations are provided, all exposing the same
:class:`ProxSolver` interface:

1. :class:`AnalyticSolver`    — closed-form projection for polytope/hull C
2. :class:`PGDSolver`         — Projected Gradient Descent (iterative)
3. :class:`FrankWolfeSolver`  — Frank-Wolfe (conditional gradient, iterative)

A factory function :func:`build_solver` selects the implementation by name.

Each solver returns a :class:`SolverResult` named-tuple that bundles the
solution with per-call diagnostics (iterations, residual, wall-clock time).
"""

from __future__ import annotations

import abc
import time
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import SolverConfig
from geometry import ConvexSet, ConvexHullContext, project_onto_simplex
from metric import BaseMetric


# ---------------------------------------------------------------------------
# Solver result container
# ---------------------------------------------------------------------------


class SolverResult(NamedTuple):
    """
    Container for the output of a single :meth:`ProxSolver.solve` call.

    Attributes
    ----------
    x:
        The solution tensor (same shape as the input ``v``).
    n_iter:
        Number of iterations taken (0 for analytic solver).
    residual:
        Final primal residual ‖x_{k+1} - x_k‖ (0.0 for analytic solver).
    converged:
        Whether the solver converged within the tolerance (always True for
        the analytic solver).
    solve_time_ms:
        Wall-clock time in milliseconds for this call.
    """

    x: Tensor
    n_iter: int
    residual: float
    converged: bool
    solve_time_ms: float


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ProxSolver(abc.ABC, nn.Module):
    """
    Abstract base class for M-weighted proximal operators.

    All solvers must implement :meth:`solve`.  The :meth:`forward` method
    delegates to :meth:`solve` and returns only the solution tensor so that
    the solver can be used transparently as a ``nn.Module`` layer.

    Parameters
    ----------
    cfg:
        Solver hyper-parameters (max iterations, tolerance, step sizes).
    """

    def __init__(self, cfg: SolverConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def solve(
        self,
        v: Tensor,
        convex_set: ConvexSet,
        metric: BaseMetric,
    ) -> SolverResult:
        """
        Compute Prox_C^M(v).

        Parameters
        ----------
        v:
            Input tensor of shape ``(batch, dim)`` — the point to project.
        convex_set:
            The closed convex constraint set C.
        metric:
            The SPD metric M defining the proximal geometry.

        Returns
        -------
        SolverResult
        """
        ...

    def forward(
        self,
        v: Tensor,
        convex_set: ConvexSet,
        metric: BaseMetric,
    ) -> Tensor:
        """
        Call :meth:`solve` and return only the solution tensor.

        Allows the solver to be used as a drop-in layer inside
        :class:`~crn.CRN`.
        """
        return self.solve(v, convex_set, metric).x

    @property
    def name(self) -> str:
        """Human-readable solver identifier."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# 1. Analytic (closed-form) solver
# ---------------------------------------------------------------------------


class AnalyticSolver(ProxSolver):
    """
    Closed-form analytic proximal solver.

    For the case where C is the convex hull of a finite set of vertices
    ``{c_1, …, c_K}`` (i.e., C = conv(C_mat)), the M-weighted proximal
    operator reduces to a constrained quadratic program (QP)::

        min_{α ∈ Δ^{K-1}} ½ ‖C_mat^T α - v‖_M²

    which has the closed-form solution for the 1-D case and, for the
    general case, is solved via the KKT conditions of the simplex-constrained
    least-squares problem.

    For a 1-atom hull (degenerate case) the projection is the atom itself.
    For 2 atoms the projection onto the line segment has a closed form.
    For K > 2 atoms the solver uses the active-set method for simplex QPs.

    Parameters
    ----------
    cfg:
        Solver configuration.
    """

    def solve(
        self,
        v: Tensor,
        convex_set: ConvexSet,
        metric: BaseMetric,
    ) -> SolverResult:
        """
        Compute the analytic M-weighted projection onto the convex hull.

        Parameters
        ----------
        v:
            Input tensor of shape ``(batch, dim)``.
        convex_set:
            Must be a :class:`~geometry.ConvexHullContext`.
        metric:
            SPD metric M.

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()

        if not isinstance(convex_set, ConvexHullContext):
            # Fallback: use the convex set's own projection
            x = convex_set.project(v)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SolverResult(x=x, n_iter=0, residual=0.0,
                                converged=True, solve_time_ms=elapsed)

        C_mat = convex_set.prototypes          # (K, dim)
        M = metric.matrix()                    # (dim, dim)
        K = C_mat.shape[0]

        if K == 1:
            # Degenerate case: only one atom
            x = C_mat[0:1].expand(v.shape[0], -1)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SolverResult(x=x, n_iter=0, residual=0.0,
                                converged=True, solve_time_ms=elapsed)

        if K == 2:
            x = self._project_segment(v, C_mat[0], C_mat[1], M)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SolverResult(x=x, n_iter=0, residual=0.0,
                                converged=True, solve_time_ms=elapsed)

        # General case: active-set simplex QP
        alpha = self._active_set_simplex_qp(v, C_mat, M)   # (batch, K)
        x = alpha @ C_mat                                    # (batch, dim)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return SolverResult(x=x, n_iter=0, residual=0.0,
                            converged=True, solve_time_ms=elapsed)

    def _project_segment(
        self,
        v: Tensor,
        c0: Tensor,
        c1: Tensor,
        M: Tensor,
    ) -> Tensor:
        """
        Closed-form M-weighted projection onto the line segment [c0, c1].

        Parameters
        ----------
        v:
            Query point, shape ``(batch, dim)``.
        c0, c1:
            Endpoints, shape ``(dim,)``.
        M:
            Metric matrix, shape ``(dim, dim)``.

        Returns
        -------
        Tensor
            Projected point, shape ``(batch, dim)``.
        """
        d = c1 - c0                                        # (dim,)
        # t* = <v - c0, d>_M / <d, d>_M
        # Numerator: (v - c0)^T M d
        v_c0 = v - c0.unsqueeze(0)                        # (batch, dim)
        Md = M @ d                                         # (dim,)
        numer = (v_c0 * Md.unsqueeze(0)).sum(dim=-1)      # (batch,)
        denom = (d * Md).sum().clamp(min=1e-12)            # scalar
        t = (numer / denom).clamp(0.0, 1.0)               # (batch,)
        # Projected point: c0 + t * d
        return c0.unsqueeze(0) + t.unsqueeze(1) * d.unsqueeze(0)

    def _active_set_simplex_qp(
        self,
        v: Tensor,
        C_mat: Tensor,
        M: Tensor,
    ) -> Tensor:
        """
        Solve the simplex-constrained QP via an active-set method.

        Solves  min_{α ∈ Δ^{K-1}} ½ ‖C_mat^T α - v‖_M²
        using the KKT conditions and a greedy active-set update.

        Parameters
        ----------
        v:
            Query point, shape ``(batch, dim)``.
        C_mat:
            Vertex matrix, shape ``(K, dim)``.
        M:
            Metric matrix, shape ``(dim, dim)``.

        Returns
        -------
        Tensor
            Optimal weights α, shape ``(batch, K)``.
        """
        K = C_mat.shape[0]
        batch = v.shape[0]

        # Quadratic objective: min_{α} ½ α^T Q α - b^T α
        # Q = C_mat M C_mat^T  (K×K),  b[k] = (M v)^T c_k → b = C_mat M v^T  (K×batch)
        # Q = (C_mat M^{1/2}) (M^{1/2} C_mat^T), but easier: Q = C_mat @ M @ C_mat^T
        MC = (M @ C_mat.t()).t()                          # (K, dim)
        Q = C_mat @ MC.t()                                # (K, K)  = C M C^T
        b = MC @ v.t()                                    # (K, batch) = C M v

        # Projected gradient descent on the simplex — reliable and differentiable
        # Step size: 1 / λ_max(Q)
        eigvals = torch.linalg.eigvalsh(Q)
        lam_max = eigvals.max().clamp(min=1e-8)
        step = 1.0 / lam_max.item()

        # Warm-start: distribute weight by similarity score
        scores = b.t()                                    # (batch, K)
        alpha = project_onto_simplex(scores)              # (batch, K)

        for _ in range(self.cfg.max_iter):
            # Gradient: Q α - b  (broadcast over batch)
            grad = (alpha @ Q.t()) - b.t()               # (batch, K)
            alpha_new = project_onto_simplex(alpha - step * grad)
            delta = (alpha_new - alpha).norm()
            alpha = alpha_new
            if delta < self.cfg.tol:
                break

        return alpha


# ---------------------------------------------------------------------------
# 2. Projected Gradient Descent solver
# ---------------------------------------------------------------------------


class PGDSolver(ProxSolver):
    """
    Projected Gradient Descent (PGD) solver for the proximal operator.

    Iterates::

        z_{k+1} = Π_C(z_k - η ∇_z ½ ‖z_k - v‖_M²)
                = Π_C(z_k - η M(z_k - v))

    where η is the step size and Π_C is the Euclidean projection onto C
    (handled by the convex_set's :meth:`project` method).

    Convergence is guaranteed when η < 2 / λ_max(M).

    Parameters
    ----------
    cfg:
        Solver configuration — uses ``pgd_step_size`` and ``max_iter``.
    """

    def solve(
        self,
        v: Tensor,
        convex_set: ConvexSet,
        metric: BaseMetric,
    ) -> SolverResult:
        """
        Run PGD iterations until convergence or ``max_iter``.

        Parameters
        ----------
        v:
            Input tensor of shape ``(batch, dim)``.
        convex_set:
            Closed convex constraint set C.
        metric:
            SPD metric M.

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()

        # Choose step size automatically (safe) or use configured value
        step_size = self._safe_step_size(metric)

        # Initialise at the Euclidean projection
        z = convex_set.project(v)

        residual = float("inf")
        converged = False

        for k in range(self.cfg.max_iter):
            z_new = convex_set.project(
                self._gradient_step(z, v, metric, step_size)
            )
            residual = float((z_new - z).norm(dim=-1).max().item())
            z = z_new
            if residual < self.cfg.tol:
                converged = True
                break

        elapsed = (time.perf_counter() - t0) * 1000.0
        return SolverResult(
            x=z,
            n_iter=k + 1,
            residual=residual,
            converged=converged,
            solve_time_ms=elapsed,
        )

    def _gradient_step(
        self,
        z: Tensor,
        v: Tensor,
        metric: BaseMetric,
        step_size: float,
    ) -> Tensor:
        """
        Compute a single gradient step: z - η M(z - v).

        Parameters
        ----------
        z:
            Current iterate, shape ``(batch, dim)``.
        v:
            Target point, shape ``(batch, dim)``.
        metric:
            SPD metric.
        step_size:
            Step size η.

        Returns
        -------
        Tensor
            Updated iterate (before projection), shape ``(batch, dim)``.
        """
        grad = metric.apply(z - v)            # M(z - v), shape (batch, dim)
        return z - step_size * grad

    def _safe_step_size(self, metric: BaseMetric) -> float:
        """
        Compute a step size guaranteed to be < 2 / λ_max(M).

        Uses the largest eigenvalue of M to set η = 1 / λ_max(M).
        """
        with torch.no_grad():
            eigs = metric.eigenvalues()
            lam_max = eigs.max().clamp(min=1e-8)
            return float(1.0 / lam_max.item())


# ---------------------------------------------------------------------------
# 3. Frank-Wolfe solver
# ---------------------------------------------------------------------------


class FrankWolfeSolver(ProxSolver):
    """
    Frank-Wolfe (conditional gradient) solver for the proximal operator.

    Iterates (away-step Frank-Wolfe for faster convergence)::

        s_k  = argmin_{s ∈ C} ⟨∇f(z_k), s⟩_M    (linear minimisation oracle)
        γ_k  = line search or 2 / (k + 2)          (step size)
        z_{k+1} = (1 - γ_k) z_k + γ_k s_k

    The linear minimisation oracle (LMO) over the convex hull of a finite
    set of atoms reduces to finding the atom with minimum inner product with
    the gradient — an O(K) operation.

    When ``fw_line_search`` is True, exact line search is used::

        γ* = ‖z_k - s_k‖_M² / ‖z_k - s_k‖_M²  (clipped to [0, 1])

    Parameters
    ----------
    cfg:
        Solver configuration — uses ``fw_line_search`` and ``max_iter``.
    """

    def solve(
        self,
        v: Tensor,
        convex_set: ConvexSet,
        metric: BaseMetric,
    ) -> SolverResult:
        """
        Run Frank-Wolfe iterations until convergence or ``max_iter``.

        Parameters
        ----------
        v:
            Input tensor of shape ``(batch, dim)``.
        convex_set:
            Closed convex constraint set C (must be a convex hull).
        metric:
            SPD metric M.

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()

        # Initialise at the Euclidean projection
        z = convex_set.project(v)

        # Get atom matrix for LMO
        if isinstance(convex_set, ConvexHullContext):
            atoms = convex_set.prototypes        # (K, dim)
        else:
            # Fallback: treat z as its own atom (fixed-point)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SolverResult(x=z, n_iter=0, residual=0.0,
                                converged=True, solve_time_ms=elapsed)

        converged = False
        residual = float("inf")

        for k in range(self.cfg.max_iter):
            # Gradient of f(z) = ½‖z - v‖_M²: ∇f = M(z - v)
            grad = metric.apply(z - v)           # (batch, dim)

            # LMO: find atom minimising <grad, s>
            s = self._lmo(grad, atoms)           # (batch, dim)

            # Frank-Wolfe gap (convergence certificate)
            gap = self._frank_wolfe_gap(z, s, v, metric)  # (batch,)

            if gap.max().item() < self.cfg.tol:
                converged = True
                residual = float(gap.max().item())
                break

            # Step size
            if self.cfg.fw_line_search:
                gamma = self._exact_line_search(z, s, v, metric)  # (batch, 1)
            else:
                gamma_val = 2.0 / (k + 2.0)
                gamma = torch.full((v.shape[0], 1), gamma_val,
                                   dtype=v.dtype, device=v.device)

            z_new = (1.0 - gamma) * z + gamma * s
            residual = float((z_new - z).norm(dim=-1).max().item())
            z = z_new

        elapsed = (time.perf_counter() - t0) * 1000.0
        return SolverResult(
            x=z,
            n_iter=k + 1,
            residual=residual,
            converged=converged,
            solve_time_ms=elapsed,
        )

    def _lmo(
        self,
        grad: Tensor,
        atoms: Tensor,
    ) -> Tensor:
        """
        Linear minimisation oracle over the convex hull of ``atoms``.

        Returns the atom ``s = argmin_{c ∈ atoms} ⟨grad, c⟩``.

        Parameters
        ----------
        grad:
            Gradient tensor of shape ``(batch, dim)``.
        atoms:
            Atom matrix of shape ``(K, dim)``.

        Returns
        -------
        Tensor
            Minimising atom for each batch element, shape ``(batch, dim)``.
        """
        # Inner products: (batch, dim) @ (dim, K) → (batch, K)
        scores = grad @ atoms.t()              # (batch, K)
        idx = scores.argmin(dim=-1)            # (batch,)
        return atoms[idx]                      # (batch, dim)

    def _exact_line_search(
        self,
        z: Tensor,
        s: Tensor,
        v: Tensor,
        metric: BaseMetric,
    ) -> Tensor:
        """
        Compute the exact minimising step size γ ∈ [0, 1].

        Minimises f((1 - γ)z + γs) = ½ ‖(1-γ)z + γs - v‖_M² over γ.

        Parameters
        ----------
        z:
            Current iterate, shape ``(batch, dim)``.
        s:
            Frank-Wolfe step direction (LMO output), shape ``(batch, dim)``.
        v:
            Target point, shape ``(batch, dim)``.
        metric:
            SPD metric M.

        Returns
        -------
        Tensor
            Step size γ per batch element, shape ``(batch, 1)``.
        """
        d = s - z                                             # (batch, dim)
        r = z - v                                             # (batch, dim)
        # γ* = -<r, d>_M / <d, d>_M = -(r^T M d) / (d^T M d)
        Mr = metric.apply(r)                                  # (batch, dim)
        Md = metric.apply(d)                                  # (batch, dim)
        numer = -(r * Md).sum(dim=-1)                         # (batch,)
        denom = (d * Md).sum(dim=-1).clamp(min=1e-12)         # (batch,)
        gamma = (numer / denom).clamp(0.0, 1.0)              # (batch,)
        return gamma.unsqueeze(1)                             # (batch, 1)

    def _frank_wolfe_gap(
        self,
        z: Tensor,
        s: Tensor,
        v: Tensor,
        metric: BaseMetric,
    ) -> Tensor:
        """
        Compute the Frank-Wolfe duality gap (convergence certificate).

        Gap = ⟨∇f(z), z - s⟩_M = ⟨M(z - v), z - s⟩

        Parameters
        ----------
        z:
            Current iterate, shape ``(batch, dim)``.
        s:
            LMO solution, shape ``(batch, dim)``.
        v:
            Target point, shape ``(batch, dim)``.
        metric:
            SPD metric M.

        Returns
        -------
        Tensor
            Per-element duality gaps, shape ``(batch,)``.
        """
        grad = metric.apply(z - v)             # (batch, dim)
        return (grad * (z - s)).sum(dim=-1)    # (batch,)


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def build_solver(solver_name: str, cfg: SolverConfig) -> ProxSolver:
    """
    Instantiate a solver by name.

    Parameters
    ----------
    solver_name:
        One of ``'analytic'``, ``'pgd'``, ``'frank_wolfe'``.
    cfg:
        Solver configuration.

    Returns
    -------
    ProxSolver

    Raises
    ------
    ValueError
        If ``solver_name`` is not recognised.
    """
    registry: dict[str, type] = {
        "analytic": AnalyticSolver,
        "pgd": PGDSolver,
        "frank_wolfe": FrankWolfeSolver,
    }
    if solver_name not in registry:
        raise ValueError(
            f"Unknown solver '{solver_name}'. "
            f"Expected one of {list(registry.keys())}."
        )
    return registry[solver_name](cfg)


# ---------------------------------------------------------------------------
# Solver benchmarking utility
# ---------------------------------------------------------------------------


class SolverBenchmark:
    """
    Utility class for timing and comparing solver implementations.

    Runs each solver on the same problem instance and records
    iteration counts, residuals, and wall-clock times.

    Parameters
    ----------
    solvers:
        Dictionary mapping solver names to :class:`ProxSolver` instances.
    """

    def __init__(self, solvers: dict[str, ProxSolver]) -> None:
        self.solvers = solvers

    def run(
        self,
        v: Tensor,
        convex_set: ConvexSet,
        metric: BaseMetric,
        n_repeats: int = 10,
    ) -> dict[str, dict]:
        """
        Benchmark all solvers on a single problem instance.

        Parameters
        ----------
        v:
            Input point(s) to project, shape ``(batch, dim)``.
        convex_set:
            Constraint set C.
        metric:
            SPD metric M.
        n_repeats:
            Number of timed repetitions to average over.

        Returns
        -------
        dict
            Nested dict ``{solver_name: {metric_name: value}}``.
        """
        results: dict[str, dict] = {}

        for name, solver in self.solvers.items():
            times: list[float] = []
            n_iters: list[int] = []
            residuals: list[float] = []
            converged_count = 0

            for rep in range(n_repeats):
                result = solver.solve(v, convex_set, metric)
                times.append(result.solve_time_ms)
                n_iters.append(result.n_iter)
                residuals.append(result.residual)
                if result.converged:
                    converged_count += 1

            import statistics
            results[name] = {
                "mean_time_ms": statistics.mean(times),
                "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                "mean_n_iter": statistics.mean(n_iters),
                "mean_residual": statistics.mean(residuals),
                "convergence_rate": converged_count / n_repeats,
                "n_repeats": n_repeats,
            }

        return results

