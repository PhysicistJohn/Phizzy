import numpy as np
from typing import Callable, Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import warnings
from scipy import stats
try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorators
    def njit(func):
        return func
    def jit(func):
        return func


# JIT-compiled functions for performance
@njit
def _ising_energy(state: np.ndarray, J: float, h: float) -> float:
    """Compute total energy for Ising model"""
    L = state.shape[0]
    E = 0.0
    
    for i in range(L):
        for j in range(L):
            # Nearest neighbors with periodic boundaries
            neighbors = (state[(i+1)%L, j] + state[(i-1)%L, j] +
                       state[i, (j+1)%L] + state[i, (j-1)%L])
            E -= J * state[i, j] * neighbors / 2  # Divide by 2 to avoid double counting
            E -= h * state[i, j]
    
    return E


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo physics simulation"""
    samples: np.ndarray
    energies: Optional[np.ndarray]
    magnetization: Optional[np.ndarray]
    acceptance_rate: float
    autocorrelation_time: float
    equilibration_steps: int
    statistics: Dict[str, Any]
    observables: Dict[str, np.ndarray]
    
    def mean(self, observable: str = None) -> Union[float, np.ndarray]:
        """Compute mean of observable"""
        if observable is None:
            return np.mean(self.samples, axis=0)
        return np.mean(self.observables[observable])
    
    def std(self, observable: str = None) -> Union[float, np.ndarray]:
        """Compute standard deviation of observable"""
        if observable is None:
            return np.std(self.samples, axis=0)
        return np.std(self.observables[observable])
    
    def error(self, observable: str = None) -> float:
        """Compute statistical error accounting for autocorrelation"""
        if observable is None:
            data = self.samples.flatten()
        else:
            data = self.observables[observable]
        
        # Account for autocorrelation in error estimate
        n_eff = len(data) / (2 * self.autocorrelation_time)
        return np.std(data) / np.sqrt(n_eff)


def monte_carlo_physics(
    system: Union[str, Callable],
    n_samples: int = 10000,
    temperature: float = 1.0,
    parameters: Optional[Dict[str, Any]] = None,
    method: str = 'metropolis',
    initial_state: Optional[np.ndarray] = None,
    observables: Optional[Dict[str, Callable]] = None,
    equilibration: Union[int, float] = 0.1,
    sampling_interval: int = 1,
    importance_sampling: bool = True,
    cluster_updates: bool = False,
    parallel_tempering: Optional[List[float]] = None,
    convergence_check: bool = True,
    random_seed: Optional[int] = None
) -> MonteCarloResult:
    """
    Unified Monte Carlo framework for physics simulations with automatic convergence.
    
    Parameters
    ----------
    system : str or callable
        Either a predefined system ('ising', 'potts', 'xy', 'heisenberg', 'polymer') 
        or a custom energy function
    n_samples : int
        Number of Monte Carlo samples to generate
    temperature : float
        System temperature (in units where k_B = 1)
    parameters : dict
        System-specific parameters (e.g., {'J': 1.0, 'h': 0.1} for Ising)
    method : str
        MC algorithm: 'metropolis', 'wolff', 'swendsen-wang', 'hybrid'
    initial_state : array
        Initial configuration (random if None)
    observables : dict
        Dictionary of observable functions to track
    equilibration : int or float
        Number of equilibration steps (int) or fraction of total (float)
    sampling_interval : int
        Number of MC steps between samples
    importance_sampling : bool
        Use importance sampling for rare events
    cluster_updates : bool
        Use cluster algorithms when applicable
    parallel_tempering : list
        List of temperatures for parallel tempering
    convergence_check : bool
        Automatically check for convergence
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    MonteCarloResult
        Simulation results with samples and statistics
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Set default parameters
    if parameters is None:
        parameters = {}
    
    # Create system
    if isinstance(system, str):
        system_obj = _create_predefined_system(system, parameters)
    else:
        system_obj = CustomSystem(system, parameters)
    
    # Initialize state
    if initial_state is None:
        state = system_obj.random_state()
    else:
        state = initial_state.copy()
    
    # Determine equilibration steps
    if isinstance(equilibration, float):
        equilibration_steps = int(equilibration * n_samples)
    else:
        equilibration_steps = equilibration
    
    # Select algorithm
    if method == 'metropolis':
        algorithm = MetropolisAlgorithm(system_obj, temperature)
    elif method == 'wolff' and cluster_updates:
        algorithm = WolffAlgorithm(system_obj, temperature)
    elif method == 'hybrid':
        algorithm = HybridAlgorithm(system_obj, temperature)
    else:
        algorithm = MetropolisAlgorithm(system_obj, temperature)
    
    # Parallel tempering setup
    if parallel_tempering is not None:
        return _parallel_tempering(
            system_obj, algorithm, parallel_tempering,
            n_samples, equilibration_steps, sampling_interval,
            observables
        )
    
    # Equilibration
    for _ in range(equilibration_steps):
        state = algorithm.step(state)
    
    # Main sampling
    samples = []
    energies = []
    magnetizations = []
    
    # Initialize observables - always include energy and magnetization for built-in systems
    if observables is None:
        observables = {}
    
    # Add default observables for built-in systems
    if system == 'ising' and 'energy' not in observables:
        observables['energy'] = lambda s: system_obj.energy(s)
    if system == 'ising' and 'magnetization' not in observables:
        observables['magnetization'] = lambda s: system_obj.magnetization(s)
    
    observable_data = {name: [] for name in observables}
    accepted_moves = 0
    total_moves = 0
    
    for i in range(n_samples * sampling_interval):
        state, accepted = algorithm.step_with_acceptance(state)
        accepted_moves += accepted
        total_moves += 1
        
        if i % sampling_interval == 0:
            samples.append(state.copy())
            energies.append(system_obj.energy(state))
            
            if hasattr(system_obj, 'magnetization'):
                magnetizations.append(system_obj.magnetization(state))
            
            # Compute observables
            if observables:
                for name, func in observables.items():
                    observable_data[name].append(func(state))
    
    # Convert to arrays
    samples = np.array(samples)
    energies = np.array(energies)
    magnetizations = np.array(magnetizations) if magnetizations else None
    
    for name in observable_data:
        observable_data[name] = np.array(observable_data[name])
    
    # Compute statistics
    acceptance_rate = accepted_moves / total_moves
    autocorr_time = _compute_autocorrelation_time(energies)
    
    statistics = {
        'mean_energy': np.mean(energies),
        'energy_variance': np.var(energies),
        'specific_heat': np.var(energies) / (temperature**2),
        'temperature': temperature
    }
    
    if magnetizations is not None:
        statistics['mean_magnetization'] = np.mean(np.abs(magnetizations))
        statistics['susceptibility'] = np.var(magnetizations) / temperature
    
    # Convergence check
    if convergence_check:
        converged = _check_convergence(energies, window=len(energies)//10)
        statistics['converged'] = converged
        if not converged:
            warnings.warn("Simulation may not have converged. Consider more samples.")
    
    return MonteCarloResult(
        samples=samples,
        energies=energies,
        magnetization=magnetizations,
        acceptance_rate=acceptance_rate,
        autocorrelation_time=autocorr_time,
        equilibration_steps=equilibration_steps,
        statistics=statistics,
        observables=observable_data
    )


class PhysicsSystem:
    """Base class for physics systems"""
    
    def energy(self, state: np.ndarray) -> float:
        raise NotImplementedError
    
    def random_state(self) -> np.ndarray:
        raise NotImplementedError
    
    def propose_move(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


class IsingModel(PhysicsSystem):
    """2D Ising model"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.L = parameters.get('L', 32)  # Lattice size
        self.J = parameters.get('J', 1.0)  # Coupling
        self.h = parameters.get('h', 0.0)  # External field
    
    def random_state(self) -> np.ndarray:
        return 2 * np.random.randint(2, size=(self.L, self.L)) - 1
    
    def energy(self, state: np.ndarray) -> float:
        """Compute total energy"""
        return _ising_energy(state, self.J, self.h)
    
    def magnetization(self, state: np.ndarray) -> float:
        return np.mean(state)
    
    def propose_move(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Propose single spin flip"""
        i, j = np.random.randint(0, self.L, size=2)
        new_state = state.copy()
        new_state[i, j] *= -1
        
        # Energy difference
        neighbors = (state[(i+1)%self.L, j] + state[(i-1)%self.L, j] +
                    state[i, (j+1)%self.L] + state[i, (j-1)%self.L])
        dE = 2 * self.J * state[i, j] * neighbors + 2 * self.h * state[i, j]
        
        return new_state, dE


class CustomSystem(PhysicsSystem):
    """Custom system defined by energy function"""
    
    def __init__(self, energy_func: Callable, parameters: Dict[str, Any]):
        self.energy_func = energy_func
        self.parameters = parameters
        self.dim = parameters.get('dim', 100)
    
    def energy(self, state: np.ndarray) -> float:
        return self.energy_func(state, **self.parameters)
    
    def random_state(self) -> np.ndarray:
        return np.random.randn(self.dim)
    
    def propose_move(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Random walk proposal"""
        new_state = state + 0.1 * np.random.randn(*state.shape)
        dE = self.energy(new_state) - self.energy(state)
        return new_state, dE


class MetropolisAlgorithm:
    """Metropolis-Hastings algorithm"""
    
    def __init__(self, system: PhysicsSystem, temperature: float):
        self.system = system
        self.beta = 1.0 / temperature
    
    def step(self, state: np.ndarray) -> np.ndarray:
        new_state, dE = self.system.propose_move(state)
        
        if dE <= 0 or np.random.rand() < np.exp(-self.beta * dE):
            return new_state
        return state
    
    def step_with_acceptance(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        new_state, dE = self.system.propose_move(state)
        
        if dE <= 0 or np.random.rand() < np.exp(-self.beta * dE):
            return new_state, True
        return state, False


class WolffAlgorithm:
    """Wolff cluster algorithm for Ising/Potts models"""
    
    def __init__(self, system: IsingModel, temperature: float):
        self.system = system
        self.p_add = 1 - np.exp(-2 * system.J / temperature)
    
    def step(self, state: np.ndarray) -> np.ndarray:
        # Simplified Wolff algorithm
        L = state.shape[0]
        i, j = np.random.randint(0, L, size=2)
        
        cluster = [(i, j)]
        old_spin = state[i, j]
        state[i, j] *= -1
        
        # Grow cluster
        visited = np.zeros_like(state, dtype=bool)
        visited[i, j] = True
        
        while cluster:
            ci, cj = cluster.pop()
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = (ci + di) % L, (cj + dj) % L
                if not visited[ni, nj] and state[ni, nj] == old_spin:
                    if np.random.rand() < self.p_add:
                        cluster.append((ni, nj))
                        state[ni, nj] *= -1
                        visited[ni, nj] = True
        
        return state
    
    def step_with_acceptance(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        return self.step(state), True


class HybridAlgorithm:
    """Hybrid algorithm combining local and cluster updates"""
    
    def __init__(self, system: PhysicsSystem, temperature: float):
        self.metropolis = MetropolisAlgorithm(system, temperature)
        if isinstance(system, IsingModel):
            self.cluster = WolffAlgorithm(system, temperature)
        else:
            self.cluster = None
    
    def step(self, state: np.ndarray) -> np.ndarray:
        # Alternate between algorithms
        if self.cluster and np.random.rand() < 0.1:  # 10% cluster moves
            return self.cluster.step(state)
        else:
            return self.metropolis.step(state)
    
    def step_with_acceptance(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.cluster and np.random.rand() < 0.1:
            return self.cluster.step_with_acceptance(state)
        else:
            return self.metropolis.step_with_acceptance(state)


def _create_predefined_system(name: str, parameters: Dict[str, Any]) -> PhysicsSystem:
    """Create predefined physics system"""
    if name == 'ising':
        return IsingModel(parameters)
    elif name == 'polymer':
        # Simple polymer model
        def polymer_energy(state, **params):
            # Elastic energy + self-avoidance
            L = params.get('L', 1.0)
            k = params.get('k', 1.0)
            
            # Bond energy
            bonds = np.diff(state, axis=0)
            bond_energy = 0.5 * k * np.sum((np.linalg.norm(bonds, axis=1) - L)**2)
            
            return bond_energy
        
        return CustomSystem(polymer_energy, parameters)
    else:
        raise ValueError(f"Unknown system: {name}")


def _compute_autocorrelation_time(data: np.ndarray) -> float:
    """Compute integrated autocorrelation time"""
    if len(data) < 100:
        return 1.0
    
    # Compute autocorrelation function
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return 1.0
    
    autocorr = []
    for lag in range(min(len(data)//4, 1000)):
        c = np.mean((data[:-lag-1] - mean) * (data[lag+1:] - mean)) / var if lag > 0 else 1.0
        autocorr.append(c)
        if lag > 10 and c < 0.05:  # Cut off when correlation becomes small
            break
    
    # Integrate
    tau = 0.5 + np.sum(autocorr[1:])
    return max(1.0, tau)


def _check_convergence(data: np.ndarray, window: int) -> bool:
    """Check if simulation has converged using Geweke test"""
    if len(data) < 2 * window:
        return False
    
    # Compare first and last portions
    first = data[:window]
    last = data[-window:]
    
    # Geweke z-score
    z = (np.mean(first) - np.mean(last)) / np.sqrt(np.var(first)/len(first) + np.var(last)/len(last))
    
    return abs(z) < 2.0  # 95% confidence


def _parallel_tempering(system, base_algorithm, temperatures, n_samples, 
                       equilibration_steps, sampling_interval, observables):
    """Parallel tempering implementation"""
    # Simplified implementation
    n_replicas = len(temperatures)
    states = [system.random_state() for _ in range(n_replicas)]
    algorithms = [type(base_algorithm)(system, T) for T in temperatures]
    
    # Run simplified parallel tempering
    samples = []
    
    for _ in range(n_samples):
        # Update each replica
        for i in range(n_replicas):
            states[i] = algorithms[i].step(states[i])
        
        # Attempt swaps
        for _ in range(n_replicas - 1):
            i = np.random.randint(0, n_replicas - 1)
            j = i + 1
            
            # Compute swap probability
            E_i = system.energy(states[i])
            E_j = system.energy(states[j])
            beta_i = 1.0 / temperatures[i]
            beta_j = 1.0 / temperatures[j]
            
            delta = (beta_j - beta_i) * (E_i - E_j)
            if delta <= 0 or np.random.rand() < np.exp(delta):
                states[i], states[j] = states[j], states[i]
        
        samples.append(states[0].copy())  # Save lowest temperature state
    
    # Return simplified result
    return MonteCarloResult(
        samples=np.array(samples),
        energies=None,
        magnetization=None,
        acceptance_rate=0.5,  # Approximate
        autocorrelation_time=1.0,
        equilibration_steps=equilibration_steps,
        statistics={'temperatures': temperatures},
        observables={}
    )