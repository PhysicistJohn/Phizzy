import pytest
import numpy as np
from phizzy.pde import coupled_pde_solve


class TestCoupledPDESolve:
    
    def test_heat_equation_1d(self):
        """Test solving 1D heat equation"""
        
        def heat_eq(t, fields, derivatives):
            """Heat equation: du/dt = D * d²u/dx²"""
            D = 0.1  # Diffusion coefficient
            u = fields['u']
            d2udx2 = derivatives['u']['dx2']
            return D * d2udx2
        
        # Initial condition: Gaussian pulse
        x = np.linspace(0, 1, 50)
        u0 = np.exp(-50 * (x - 0.5)**2)
        
        initial_conditions = {'u': u0}
        boundary_conditions = {'u': {'left': 0, 'right': 0}}
        domain = {'x': (0, 1)}
        time_span = (0, 0.1)
        
        result = coupled_pde_solve(
            equations=[heat_eq],
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            domain=domain,
            time_span=time_span,
            grid_size=50
        )
        
        # Check that solution exists and is reasonable
        assert result.convergence_info['success']
        assert 'u' in result.solution_fields
        assert result.solution_fields['u'].shape[0] == 50  # Grid points
        
        # Heat equation should diffuse the initial pulse (final max < initial max)
        final_field = result.get_field('u', -1)
        assert np.max(final_field) < np.max(u0)
    
    def test_wave_equation_1d(self):
        """Test solving 1D wave equation as coupled system"""
        
        def wave_eq_u(t, fields, derivatives):
            """u equation: du/dt = v"""
            return fields['v']
        
        def wave_eq_v(t, fields, derivatives):
            """v equation: dv/dt = c² * d²u/dx²"""
            c = 1.0  # Wave speed
            d2udx2 = derivatives['u']['dx2']
            return c**2 * d2udx2
        
        # Initial conditions: Gaussian pulse at rest
        x = np.linspace(0, 2, 40)
        u0 = np.exp(-50 * (x - 1.0)**2)
        v0 = np.zeros_like(x)
        
        initial_conditions = {'u': u0, 'v': v0}
        boundary_conditions = {
            'u': {'left': 0, 'right': 0},
            'v': {'left': 0, 'right': 0}
        }
        domain = {'x': (0, 2)}
        time_span = (0, 0.5)
        
        result = coupled_pde_solve(
            equations=[wave_eq_u, wave_eq_v],
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            domain=domain,
            time_span=time_span,
            grid_size=40
        )
        
        assert result.convergence_info['success']
        assert 'u' in result.solution_fields
        assert 'v' in result.solution_fields
        
        # Wave should propagate (conservation of energy)
        final_u = result.get_field('u', -1)
        final_v = result.get_field('v', -1)
        
        # Total energy should be roughly conserved
        initial_energy = np.sum(u0**2) + np.sum(v0**2)
        final_energy = np.sum(final_u**2) + np.sum(final_v**2)
        
        # Allow for some numerical dissipation
        assert final_energy > 0.5 * initial_energy
    
    def test_reaction_diffusion_system(self):
        """Test coupled reaction-diffusion system"""
        
        def reaction_diffusion_u(t, fields, derivatives):
            """du/dt = D_u * d²u/dx² + a*u - b*u*v"""
            D_u = 0.1
            a, b = 1.0, 2.0
            u = fields['u']
            v = fields['v']
            d2udx2 = derivatives['u']['dx2']
            return D_u * d2udx2 + a*u - b*u*v
        
        def reaction_diffusion_v(t, fields, derivatives):
            """dv/dt = D_v * d²v/dx² + b*u*v - c*v"""
            D_v = 0.05
            b, c = 2.0, 1.5
            u = fields['u']
            v = fields['v']
            d2vdx2 = derivatives['v']['dx2']
            return D_v * d2vdx2 + b*u*v - c*v
        
        # Initial conditions: random perturbations around steady state
        x = np.linspace(0, 1, 30)
        u0 = 0.5 + 0.1 * np.random.randn(30)
        v0 = 0.3 + 0.1 * np.random.randn(30)
        
        initial_conditions = {'u': u0, 'v': v0}
        boundary_conditions = {
            'u': {'left_neumann': True, 'right_neumann': True},
            'v': {'left_neumann': True, 'right_neumann': True}
        }
        domain = {'x': (0, 1)}
        time_span = (0, 1.0)
        
        result = coupled_pde_solve(
            equations=[reaction_diffusion_u, reaction_diffusion_v],
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            domain=domain,
            time_span=time_span,
            grid_size=30
        )
        
        assert result.convergence_info['success']
        assert 'u' in result.solution_fields
        assert 'v' in result.solution_fields
        
        # Solution should remain bounded for this system
        final_u = result.get_field('u', -1)
        final_v = result.get_field('v', -1)
        
        assert np.all(np.isfinite(final_u))
        assert np.all(np.isfinite(final_v))
        assert np.all(final_u >= 0)  # Concentrations should be non-negative
        assert np.all(final_v >= 0)
    
    def test_boundary_conditions(self):
        """Test different boundary condition types"""
        
        def simple_diffusion(t, fields, derivatives):
            return 0.1 * derivatives['u']['dx2']
        
        x = np.linspace(0, 1, 20)
        u0 = np.sin(np.pi * x)
        
        # Test Dirichlet boundary conditions
        initial_conditions = {'u': u0}
        boundary_conditions = {'u': {'left': 0, 'right': 0}}
        domain = {'x': (0, 1)}
        time_span = (0, 0.1)
        
        result = coupled_pde_solve(
            equations=[simple_diffusion],
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            domain=domain,
            time_span=time_span,
            grid_size=20
        )
        
        assert result.convergence_info['success']
        
        # Check boundary conditions are enforced
        final_field = result.get_field('u', -1)
        assert abs(final_field[0]) < 1e-10  # Left boundary
        assert abs(final_field[-1]) < 1e-10  # Right boundary
    
    def test_pde_result_methods(self):
        """Test PDEResult methods and properties"""
        
        def dummy_eq(t, fields, derivatives):
            return np.zeros_like(fields['u'])
        
        x = np.linspace(0, 1, 10)
        u0 = np.ones(10)
        
        initial_conditions = {'u': u0}
        boundary_conditions = {'u': {'left': 1, 'right': 1}}
        domain = {'x': (0, 1)}
        time_span = (0, 0.1)
        
        result = coupled_pde_solve(
            equations=[dummy_eq],
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            domain=domain,
            time_span=time_span,
            grid_size=10
        )
        
        # Test get_field method
        field_at_start = result.get_field('u', 0)
        field_at_end = result.get_field('u', -1)
        
        assert len(field_at_start) == 10
        assert len(field_at_end) == 10
        
        # Test error handling
        with pytest.raises(ValueError):
            result.get_field('nonexistent_field')
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        
        def unstable_eq(t, fields, derivatives):
            # Deliberately unstable equation
            return 100 * fields['u']
        
        x = np.linspace(0, 1, 10)
        u0 = np.ones(10)
        
        initial_conditions = {'u': u0}
        boundary_conditions = {'u': {'left': 1, 'right': 1}}
        domain = {'x': (0, 1)}
        time_span = (0, 10)  # Long time span
        
        # This should either fail gracefully or indicate lack of success
        result = coupled_pde_solve(
            equations=[unstable_eq],
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            domain=domain,
            time_span=time_span,
            grid_size=10,
            tolerance=1e-6
        )
        
        # Should have convergence information regardless of success
        assert 'success' in result.convergence_info
        assert 'message' in result.convergence_info