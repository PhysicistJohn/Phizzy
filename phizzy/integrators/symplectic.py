import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SymplecticResult:
    """Result of symplectic integration"""
    t: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray
    energy_error: float
    info: Dict[str, Any]


def symplectic_integrate(
    hamiltonian: Callable[[np.ndarray, np.ndarray], float],
    q0: np.ndarray,
    p0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    method: str = 'leapfrog',
    energy_tol: float = 1e-10,
    max_energy_error: float = 1e-6,
    callback: Optional[Callable] = None
) -> SymplecticResult:
    """
    Symplectic integrator for Hamiltonian systems that preserves phase space structure.
    
    Parameters
    ----------
    hamiltonian : callable
        Hamiltonian function H(q, p) where q is position and p is momentum
    q0 : np.ndarray
        Initial position(s)
    p0 : np.ndarray
        Initial momentum/momenta
    t_span : tuple
        Time interval (t0, tf)
    dt : float
        Time step size
    method : str
        Integration method: 'leapfrog', 'yoshida4', 'forest-ruth', 'pefrl'
    energy_tol : float
        Tolerance for energy conservation check
    max_energy_error : float
        Maximum allowed relative energy error
    callback : callable, optional
        Function called at each time step
        
    Returns
    -------
    SymplecticResult
        Object containing time points, positions, momenta, energies, and diagnostics
    """
    
    q0 = np.atleast_1d(q0).astype(float)
    p0 = np.atleast_1d(p0).astype(float)
    
    if q0.shape != p0.shape:
        raise ValueError("q0 and p0 must have the same shape")
    
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n_steps + 1)
    
    # Storage
    q_hist = np.zeros((n_steps + 1, *q0.shape))
    p_hist = np.zeros((n_steps + 1, *p0.shape))
    energy_hist = np.zeros(n_steps + 1)
    
    # Initial conditions
    q_hist[0] = q0
    p_hist[0] = p0
    energy_hist[0] = hamiltonian(q0, p0)
    initial_energy = energy_hist[0]
    
    # Choose integrator
    if method == 'leapfrog':
        integrator = _leapfrog_step
    elif method == 'yoshida4':
        integrator = _yoshida4_step
    elif method == 'forest-ruth':
        integrator = _forest_ruth_step
    elif method == 'pefrl':
        integrator = _pefrl_step
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Integration loop
    q, p = q0.copy(), p0.copy()
    max_energy_deviation = 0.0
    
    for i in range(n_steps):
        # Take a step
        q, p = integrator(hamiltonian, q, p, dt)
        
        # Store results
        q_hist[i + 1] = q
        p_hist[i + 1] = p
        energy_hist[i + 1] = hamiltonian(q, p)
        
        # Check energy conservation
        energy_error = abs(energy_hist[i + 1] - initial_energy) / abs(initial_energy)
        max_energy_deviation = max(max_energy_deviation, energy_error)
        
        if energy_error > max_energy_error:
            import warnings
            warnings.warn(
                f"Large energy error at t={t[i+1]:.3f}: {energy_error:.2e} > {max_energy_error:.2e}"
            )
        
        # Callback
        if callback is not None:
            callback(t[i + 1], q, p)
    
    return SymplecticResult(
        t=t,
        q=q_hist,
        p=p_hist,
        energy=energy_hist,
        energy_error=max_energy_deviation,
        info={
            'method': method,
            'dt': dt,
            'n_steps': n_steps,
            'energy_drift': energy_hist[-1] - initial_energy,
            'relative_energy_drift': (energy_hist[-1] - initial_energy) / abs(initial_energy)
        }
    )


def _compute_derivatives(hamiltonian: Callable, q: np.ndarray, p: np.ndarray, 
                        eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dH/dq and dH/dp using finite differences"""
    dq = np.zeros_like(q)
    dp = np.zeros_like(p)
    
    # dH/dq
    for i in range(len(q)):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += eps
        q_minus[i] -= eps
        dq[i] = (hamiltonian(q_plus, p) - hamiltonian(q_minus, p)) / (2 * eps)
    
    # dH/dp
    for i in range(len(p)):
        p_plus = p.copy()
        p_minus = p.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        dp[i] = (hamiltonian(q, p_plus) - hamiltonian(q, p_minus)) / (2 * eps)
    
    return dq, dp


def _leapfrog_step(hamiltonian: Callable, q: np.ndarray, p: np.ndarray, 
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Standard leapfrog/Störmer-Verlet integrator (2nd order)"""
    dH_dq, dH_dp = _compute_derivatives(hamiltonian, q, p)
    
    # Half step for momentum
    p_half = p - 0.5 * dt * dH_dq
    
    # Full step for position
    q_new = q + dt * dH_dp
    
    # Update derivatives at new position
    dH_dq_new, _ = _compute_derivatives(hamiltonian, q_new, p_half)
    
    # Half step for momentum
    p_new = p_half - 0.5 * dt * dH_dq_new
    
    return q_new, p_new


def _yoshida4_step(hamiltonian: Callable, q: np.ndarray, p: np.ndarray, 
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Yoshida 4th order symplectic integrator"""
    # Yoshida coefficients
    w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    w0 = -2.0**(1.0/3.0) * w1
    c1 = w1 / 2.0
    c2 = (w0 + w1) / 2.0
    c3 = c2
    c4 = c1
    d1 = w1
    d2 = w0
    d3 = w1
    
    # Apply composition
    q1, p1 = _leapfrog_substep(hamiltonian, q, p, c1 * dt, d1 * dt)
    q2, p2 = _leapfrog_substep(hamiltonian, q1, p1, c2 * dt, d2 * dt)
    q3, p3 = _leapfrog_substep(hamiltonian, q2, p2, c3 * dt, d3 * dt)
    q4, p4 = _leapfrog_substep(hamiltonian, q3, p3, c4 * dt, 0)
    
    return q4, p4


def _forest_ruth_step(hamiltonian: Callable, q: np.ndarray, p: np.ndarray, 
                      dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Forest-Ruth 4th order symplectic integrator"""
    theta = 1.0 / (2.0 - 2.0**(1.0/3.0))
    
    c1 = theta / 2.0
    c2 = (1.0 - theta) / 2.0
    c3 = c2
    c4 = c1
    d1 = theta
    d2 = -theta
    d3 = 1.0 - 2.0 * theta
    d4 = theta
    
    q1, p1 = _leapfrog_substep(hamiltonian, q, p, c1 * dt, d1 * dt)
    q2, p2 = _leapfrog_substep(hamiltonian, q1, p1, c2 * dt, d2 * dt)
    q3, p3 = _leapfrog_substep(hamiltonian, q2, p2, c3 * dt, d3 * dt)
    q4, p4 = _leapfrog_substep(hamiltonian, q3, p3, c4 * dt, d4 * dt)
    
    return q4, p4


def _pefrl_step(hamiltonian: Callable, q: np.ndarray, p: np.ndarray, 
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Position Extended Forest-Ruth Like (PEFRL) integrator (4th order)"""
    xi = 0.1786178958448091
    lambda_ = -0.2123418310626054
    chi = -0.06626458266981849
    
    c1 = xi
    c2 = chi
    c3 = 1.0 - 2.0 * (chi + xi)
    c4 = chi
    c5 = xi
    d1 = 2.0 * xi
    d2 = 1.0 - 4.0 * xi - 2.0 * lambda_
    d3 = 2.0 * lambda_
    d4 = d2
    d5 = d1
    
    q1, p1 = _leapfrog_substep(hamiltonian, q, p, c1 * dt, d1 * dt)
    q2, p2 = _leapfrog_substep(hamiltonian, q1, p1, c2 * dt, d2 * dt)
    q3, p3 = _leapfrog_substep(hamiltonian, q2, p2, c3 * dt, d3 * dt)
    q4, p4 = _leapfrog_substep(hamiltonian, q3, p3, c4 * dt, d4 * dt)
    q5, p5 = _leapfrog_substep(hamiltonian, q4, p4, c5 * dt, d5 * dt / 2.0)
    
    return q5, p5


def _leapfrog_substep(hamiltonian: Callable, q: np.ndarray, p: np.ndarray,
                      c_dt: float, d_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Helper for composed methods"""
    if c_dt != 0:
        dH_dq, _ = _compute_derivatives(hamiltonian, q, p)
        p = p - c_dt * dH_dq
    
    if d_dt != 0:
        _, dH_dp = _compute_derivatives(hamiltonian, q, p)
        q = q + d_dt * dH_dp
    
    return q, p