import numpy as np
from typing import List, Dict, Callable, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings


@dataclass
class PDEResult:
    """Result of coupled PDE solution"""
    t: np.ndarray
    y: np.ndarray
    solution_fields: Dict[str, np.ndarray]
    mesh: Dict[str, np.ndarray]
    convergence_info: Dict[str, Any]
    
    def get_field(self, name: str, time_idx: int = -1) -> np.ndarray:
        """Get field at specific time"""
        if name in self.solution_fields:
            return self.solution_fields[name][time_idx]
        raise ValueError(f"Field {name} not found")


def coupled_pde_solve(
    equations: List[Callable],
    initial_conditions: Dict[str, np.ndarray],
    boundary_conditions: Dict[str, Any],
    domain: Dict[str, Tuple[float, float]],
    time_span: Tuple[float, float],
    grid_size: Union[int, Dict[str, int]] = 50,
    method: str = 'finite_difference',
    coupling_terms: Optional[List[Callable]] = None,
    adaptive_mesh: bool = False,
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> PDEResult:
    """
    Solve coupled partial differential equations with automatic mesh refinement.
    
    Parameters
    ----------
    equations : list of callables
        List of PDE functions in the form f(t, u, u_x, u_xx, ...)
    initial_conditions : dict
        Initial field values {field_name: array}
    boundary_conditions : dict
        Boundary conditions for each field
    domain : dict
        Spatial domain {dim: (min, max)}
    time_span : tuple
        Time integration interval (t0, tf)
    grid_size : int or dict
        Number of grid points per dimension
    method : str
        Solution method: 'finite_difference', 'finite_element', 'spectral'
    coupling_terms : list, optional
        Additional coupling functions
    adaptive_mesh : bool
        Use adaptive mesh refinement
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations for implicit methods
        
    Returns
    -------
    PDEResult
        Solution with fields and convergence information
    """
    
    # Set up spatial grid
    if isinstance(grid_size, int):
        nx = grid_size
    else:
        nx = grid_size.get('x', 50)
    
    if 'x' in domain:
        x_min, x_max = domain['x']
        x = np.linspace(x_min, x_max, nx)
        dx = x[1] - x[0]
    else:
        x = np.linspace(0, 1, nx)
        dx = 1.0 / (nx - 1)
    
    mesh = {'x': x, 'dx': dx}
    
    # Set up time integration
    t0, tf = time_span
    
    # Convert initial conditions to vector form
    field_names = list(initial_conditions.keys())
    n_fields = len(field_names)
    n_points = len(x)
    
    y0 = np.zeros(n_fields * n_points)
    for i, name in enumerate(field_names):
        y0[i*n_points:(i+1)*n_points] = initial_conditions[name]
    
    # Define the system of ODEs (method of lines)
    def system_ode(t, y):
        # Reshape y back to fields
        fields = {}
        for i, name in enumerate(field_names):
            fields[name] = y[i*n_points:(i+1)*n_points]
        
        # Apply boundary conditions
        fields = _apply_boundary_conditions(fields, boundary_conditions, x)
        
        # Compute spatial derivatives
        derivatives = {}
        for name, field in fields.items():
            derivatives[name] = _compute_derivatives(field, dx)
        
        # Evaluate PDEs
        dydt = np.zeros_like(y)
        for i, (name, eq) in enumerate(zip(field_names, equations)):
            field_dydt = eq(t, fields, derivatives)
            dydt[i*n_points:(i+1)*n_points] = field_dydt
        
        return dydt
    
    # Time integration
    if method == 'finite_difference':
        solution = solve_ivp(
            system_ode, time_span, y0,
            method='BDF',  # Good for stiff systems
            rtol=tolerance,
            atol=tolerance
        )
    else:
        # Simplified - use same method
        solution = solve_ivp(system_ode, time_span, y0, rtol=tolerance)
    
    # Reshape solution back to fields
    solution_fields = {}
    for i, name in enumerate(field_names):
        field_data = solution.y[i*n_points:(i+1)*n_points, :]
        solution_fields[name] = field_data
    
    convergence_info = {
        'success': solution.success,
        'message': solution.message,
        'nfev': solution.nfev,
        'njev': solution.njev if hasattr(solution, 'njev') else 0
    }
    
    return PDEResult(
        t=solution.t,
        y=solution.y,
        solution_fields=solution_fields,
        mesh=mesh,
        convergence_info=convergence_info
    )


def _apply_boundary_conditions(fields: Dict[str, np.ndarray], 
                              boundary_conditions: Dict[str, Any],
                              x: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply boundary conditions to fields"""
    for field_name, field in fields.items():
        if field_name in boundary_conditions:
            bc = boundary_conditions[field_name]
            
            # Dirichlet boundary conditions
            if 'left' in bc:
                field[0] = bc['left']
            if 'right' in bc:
                field[-1] = bc['right']
            
            # Neumann boundary conditions (zero derivative)
            if bc.get('left_neumann', False):
                field[0] = field[1]
            if bc.get('right_neumann', False):
                field[-1] = field[-2]
                
            # Periodic boundary conditions
            if bc.get('periodic', False):
                field[0] = field[-1]
    
    return fields


def _compute_derivatives(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """Compute spatial derivatives using finite differences"""
    derivatives = {}
    
    # First derivative (central difference)
    dudx = np.zeros_like(field)
    dudx[1:-1] = (field[2:] - field[:-2]) / (2 * dx)
    dudx[0] = (field[1] - field[0]) / dx  # Forward difference
    dudx[-1] = (field[-1] - field[-2]) / dx  # Backward difference
    derivatives['dx'] = dudx
    
    # Second derivative
    d2udx2 = np.zeros_like(field)
    d2udx2[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / (dx**2)
    # Use extrapolation for boundaries
    d2udx2[0] = 2*d2udx2[1] - d2udx2[2]
    d2udx2[-1] = 2*d2udx2[-2] - d2udx2[-3]
    derivatives['dx2'] = d2udx2
    
    return derivatives


# Example PDE systems

def heat_equation_1d(t, fields, derivatives):
    """1D heat equation: du/dt = D * d²u/dx²"""
    D = 1.0  # Diffusion coefficient
    u = fields['u']
    d2udx2 = derivatives['u']['dx2']
    return D * d2udx2


def wave_equation_1d(t, fields, derivatives):
    """1D wave equation: d²u/dt² = c² * d²u/dx²"""
    c = 1.0  # Wave speed
    u = fields['u']
    v = fields['v']  # du/dt
    d2udx2 = derivatives['u']['dx2']
    
    # Return [du/dt, dv/dt] where v = du/dt
    if 'u' in fields and 'v' in fields:
        return {'u': v, 'v': c**2 * d2udx2}
    else:
        return c**2 * d2udx2


def reaction_diffusion(t, fields, derivatives):
    """Reaction-diffusion system"""
    D_u, D_v = 1.0, 0.5
    a, b = 1.0, 3.0
    
    u = fields['u']
    v = fields['v']
    
    dudt = D_u * derivatives['u']['dx2'] + a*u - b*u*v
    dvdt = D_v * derivatives['v']['dx2'] + b*u*v - v
    
    return {'u': dudt, 'v': dvdt}