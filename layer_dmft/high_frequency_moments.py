r"""Implementation of high-frequency moments.

It is assumed that odd powers of epsilon vanishes due to symmetry.

Glossary
--------
eps_{n}, ϵ_{n}
   The n-th epsilon moment of the DOS defined by :math:`\int dϵ DOS(ϵ) ϵ^n`
gf, G
   Green's function :math:`G`
hybridization, Δ
   The hybridization function :math:`Δ(z) = z + μ - Σ(z) - 1/G(z)`
m{n}
   n-th high-frequency moment, the term proportional to math:`z^{-m}`
m{n}_subtract
   Given for Green's functions. The n-th high-frequency moment,
   with the n-th moment of the self-energy subtracted.
   Relevant for the moments of the hybridization function.
self, Σ
   The self-energy :math:`Σ(z) = G_0^{-1}(z) - G^{-1}(z)`
gf_x_self, G×Σ
    The product of Green's function and self-energy :math:`G(z)Σ(z)`.
    The convolution in imaginary time can be measured.

TODO
----
Use `scipy`/`numpy` polynomials or `sympy` to derive the formulas rigorously.
Attention has to be paid to the fact, that we have matrices for the lattice
Green's function, thus the variables do *not commute*.

"""
import numpy as np


def self_m0(U, occ):
    """Static part of the self-energy.

    This is the zeroth high-frequency moment :math:`Σ^{(0)}`.

    Parameters
    ----------
    U : float
        On-site interacting strength (Hubbard U)
    occ : (2, ) float np.ndarray
        Occupation numbers of the opposite spin channel

    Returns
    -------
    Σ_m0 : (2, ) float np.ndarray
        The static part of the self-energy.

    """
    occ_other = occ
    return U*occ_other


def self_m1(U, occ):
    """First high-frequency moment of the self-energy :math:`Σ^{(1)}/z`.

    Parameters
    ----------
    U : float
        On-site interacting strength (Hubbard U)
    occ : (2, ) float np.ndarray
        Occupation numbers of the opposite spin channel

    Returns
    -------
    Σ_m1 : (2, ) float np.ndarray
        The high-frequency moment

    """
    occ_other = occ  # other spin channel is relevant
    return U**2 * occ_other * (1 - occ_other)


def gf_lattice_m2(self_mod_m0):
    """Second high-frequency moment of the Green's function :math:`G^{(2)}/z^2`.

    Parameters
    ----------
    self_mod_m0
        The *modified* zeroth moment (static part) of the self-energy.
        Modified means here, that it incorporates the single particle Hamiltonian
        (on-site energies and hopping).

    Returns
    -------
    G_m2
        The high-frequency moment

    """
    return self_mod_m0


def gf_lattice_m3_subtract(self_mod_m0, eps_2):
    """Third high-frequency moment without self-energy moment :math:`(G^{(3)} - Σ^{(1)})/z^3`.

    Parameters
    ----------
    self_mod_m0 : (..., N, N) float np.ndarray
        The *modified* zeroth moment (static part) of the self-energy.
        Modified means here, that it incorporates the single particle Hamiltonian
        (on-site energies and hopping).
    eps_2 : float
        The second epsilon moment of the DOS

    Returns
    -------
    G_m3 - Σ_m1 : (..., N, N) float np.ndarray
        The high-frequency moment without leading self-energy term.

    """
    idx = np.eye(*self_mod_m0.shape[-2:])
    return self_mod_m0@self_mod_m0 + eps_2*idx


def gf_lattice_m3(self_mod_m0, self_m1, eps_2):
    """Third high-frequency moment of the Green's function :math:`G^{(3)}/z^3`.

    Parameters
    ----------
    self_mod_m0 : (..., N, N) float np.ndarray
        The *modified* zeroth moment (static part) of the self-energy.
        Modified means here, that it incorporates the single particle Hamiltonian
        (on-site energies and hopping).
    self_m1 : (..., N, N) float np.ndarray
        The first high-frequency moment of the self-energy.
    eps_2 : float
        The second epsilon moment of the DOS

    Returns
    -------
    G_m3 : (..., N, N) float np.ndarray
        The high-frequency moment

    """
    return self_m1 + gf_lattice_m3_subtract(self_mod_m0, eps_2)


def gf_lattice_m4_subtract(self_mod_m0, self_m1, eps_2):
    """Fourth high-frequency moment without self-energy moment :math:`(G^{(4)} - Σ^{(2)})/z^4`.

    Parameters
    ----------
    self_mod_m0 : (..., N, N) float np.ndarray
        The *modified* zeroth moment (static part) of the self-energy.
        Modified means here, that it incorporates the single particle Hamiltonian
        (on-site energies and hopping).
    self_m1 : (..., N, N) float np.ndarray
        The first high-frequency moment of the self-energy.
    eps_2 : float
        The second epsilon moment of the DOS

    Returns
    -------
    G_m4 - Σ_m2 : (..., N, N) float np.ndarray
        The high-frequency moment

    """
    self_01 = self_mod_m0@self_m1 + self_m1@self_mod_m0
    eps_2_self_0 = 3*eps_2*self_mod_m0
    matrix_power = np.vectorize(np.linalg.matrix_power, signature='(n,n),()->(n,n)')
    self_1_pow3 = matrix_power(self_mod_m0, 3)
    return self_01 + eps_2_self_0 + self_1_pow3


def hybridization_m1(gf_m2, gf_m3_sub):
    """First high-frequency moment of the hybridization function :math:`Δ^{(1)}/z`.

    Parameters
    ----------
    gf_m2
        Diagonal part of the second Green's function moment.
    gf_m3_sub
        Diagonal part of the third Green's function moment without leading self-energy
        term.

    Returns
    -------
    Δ_m1
        The high-frequency moment

    """
    return gf_m3_sub - gf_m2**2


def hybridization_m2(gf_m2, gf_m3, gf_m4_sub):
    """Second high-frequency moment of the hybridization function :math:`Δ^{(2)}/z^2`.

    Parameters
    ----------
    gf_m2
        Diagonal part of the second Green's function moment.
    gf_m3
        Diagonal part of the third Green's function moment.
    gf_m4_sub
        Diagonal part of the fourth Green's function moment without leading self-energy is

    Returns
    -------
    Δ_m2
        The high-frequency moment

    """
    return gf_m4_sub + gf_m2**3 - 2*gf_m3*gf_m2


def gf_x_self_m1(self_m0):
    """Fist high-frequency moment of the product of Green's function and self-energy.

    Parameters
    ----------
    self_m0
        The static part (zeroth high-frequency moment) of the self-energy.

    Returns
    -------
    G×Σ_m1
        The high-frequency moment

    """
    return self_m0


def gf_x_self_m2(self_m0, self_m1, gf_m2):
    """Second high-frequency moment of the product of Green's function and self-energy.

    This function uses that G_m1 ≡ 1.

    Parameters
    ----------
    self_m0
        The static part (zeroth high-frequency moment) of the self-energy.
    self_m1
        The first high-frequency moment of the self-energy.
    gf_m2
        The second high-frequency moment of the Green's function.

    Returns
    -------
    G×Σ_m2
        The high-frequency moment

    """
    return self_m0*gf_m2 + self_m1
