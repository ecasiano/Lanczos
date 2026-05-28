"""
Lanczos Eigensolver
===================

Wrapper around scipy.sparse.linalg.eigsh, which implements the
implicitly restarted Lanczos algorithm (via ARPACK) for finding
the lowest eigenvalues and eigenvectors of large, sparse, real
symmetric (Hermitian) matrices.

The Lanczos algorithm builds a Krylov subspace:

    K_m = span{ v, Hv, H^2 v, ..., H^{m-1} v }

and projects the Hamiltonian onto this subspace, producing a
tridiagonal matrix whose eigenvalues approximate those of H.
ARPACK adds implicit restarts to improve convergence.

For small matrices (dim <= 16), we fall back to full (dense)
diagonalization via numpy.linalg.eigh, since ARPACK requires
k < dim - 1.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
from typing import Optional


class LanczosSolver:
    """Solve for low-lying eigenvalues and eigenvectors of a sparse Hamiltonian.

    Parameters
    ----------
    hamiltonian : scipy.sparse matrix
        The Hamiltonian matrix (must be Hermitian / real symmetric).
    num_eigenvalues : int
        Number of lowest eigenvalues to compute (default 1).
    tolerance : float
        Convergence tolerance for ARPACK (default 0 = machine precision).
    max_iterations : int or None
        Maximum number of Lanczos iterations before giving up.
    """

    def __init__(self, hamiltonian: sparse.spmatrix,
                 num_eigenvalues: int = 1,
                 tolerance: float = 0.0,
                 max_iterations: Optional[int] = None):

        self.hamiltonian = hamiltonian
        self.num_eigenvalues = num_eigenvalues
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Results (populated after solve() is called)
        self._eigenvalues = None
        self._eigenvectors = None

    def solve(self):
        """Run the Lanczos diagonalization.

        For very small Hilbert spaces (dim <= 16), uses full dense
        diagonalization instead of the iterative Lanczos method.

        Returns
        -------
        eigenvalues : ndarray of shape (num_eigenvalues,)
            The lowest eigenvalues, sorted in ascending order.
        eigenvectors : ndarray of shape (dim, num_eigenvalues)
            Corresponding eigenvectors as column vectors.
        """
        hilbert_dim = self.hamiltonian.shape[0]

        # --- Edge case: trivial Hilbert space ---
        if hilbert_dim <= 1:
            if hilbert_dim == 1:
                self._eigenvalues = np.array([self.hamiltonian[0, 0]])
                self._eigenvectors = np.array([[1.0]])
            else:
                self._eigenvalues = np.array([])
                self._eigenvectors = np.array([[]])
            return self._eigenvalues, self._eigenvectors

        # --- Small matrix: use full (dense) diagonalization ---
        if hilbert_dim <= 16 or self.num_eigenvalues >= hilbert_dim - 1:
            hamiltonian_dense = (
                self.hamiltonian.toarray()
                if sparse.issparse(self.hamiltonian)
                else self.hamiltonian
            )
            all_eigenvalues, all_eigenvectors = np.linalg.eigh(hamiltonian_dense)

            num_to_keep = min(self.num_eigenvalues, hilbert_dim)
            self._eigenvalues = all_eigenvalues[:num_to_keep]
            self._eigenvectors = all_eigenvectors[:, :num_to_keep]
            return self._eigenvalues, self._eigenvectors

        # --- Large matrix: iterative Lanczos via ARPACK ---
        #
        # ARPACK (Fortran) uses 32-bit integers internally.
        # The default maxiter = dim × 10 overflows int32 when
        # dim > ~214 million (e.g. L=N=16 has dim ≈ 601M).
        # Cap at a safe value; ARPACK typically converges in
        # O(100–1000) iterations for the ground state anyway.
        _ARPACK_MAXITER_CAP = 1_000_000

        if self.max_iterations is not None:
            safe_maxiter = min(self.max_iterations, _ARPACK_MAXITER_CAP)
        else:
            safe_maxiter = min(hilbert_dim * 10, _ARPACK_MAXITER_CAP)

        arpack_options = dict(
            k=self.num_eigenvalues,
            which='SA',       # Smallest Algebraic eigenvalues
            tol=self.tolerance,
            maxiter=safe_maxiter,
        )

        eigenvalues, eigenvectors = eigsh(self.hamiltonian, **arpack_options)

        # Sort by eigenvalue (ARPACK doesn't guarantee ordering)
        sort_order = np.argsort(eigenvalues)
        self._eigenvalues = eigenvalues[sort_order]
        self._eigenvectors = eigenvectors[:, sort_order]

        return self._eigenvalues, self._eigenvectors

    @property
    def eigenvalues(self):
        """Lowest eigenvalues (computed on first access)."""
        if self._eigenvalues is None:
            self.solve()
        return self._eigenvalues

    @property
    def eigenvectors(self):
        """Eigenvectors corresponding to lowest eigenvalues."""
        if self._eigenvectors is None:
            self.solve()
        return self._eigenvectors

    @property
    def ground_state_energy(self) -> float:
        """Ground state energy E_0."""
        return self.eigenvalues[0]

    @property
    def ground_state(self) -> np.ndarray:
        """Ground state wavefunction |psi_0> as a vector in the Fock basis."""
        return self.eigenvectors[:, 0]
