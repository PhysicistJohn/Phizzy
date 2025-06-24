import pytest
import numpy as np
import matplotlib.pyplot as plt
from phizzy.visualization import field_visualization_3d


class TestFieldVisualization3D:
    
    def test_vector_field_visualization(self):
        """Test basic vector field visualization"""
        # Simple radial field
        def radial_field(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
            return [x/r, y/r, z/r]
        
        result = field_visualization_3d(
            radial_field,
            bounds=((-2, 2), (-2, 2), (-2, 2)),
            field_type='vector',
            resolution=10,
            backend='matplotlib'
        )
        
        assert result.field_type == 'vector'
        assert isinstance(result.figure, plt.Figure)
        assert 'max_magnitude' in result.statistics
        plt.close(result.figure)
    
    def test_scalar_field_visualization(self):
        """Test scalar field visualization"""
        # Gaussian blob
        def gaussian_field(x, y, z):
            return np.exp(-(x**2 + y**2 + z**2))
        
        result = field_visualization_3d(
            gaussian_field,
            bounds={'x': (-3, 3), 'y': (-3, 3), 'z': (-3, 3)},
            field_type='scalar',
            resolution=15,
            isosurfaces=[0.5, 0.1],
            backend='matplotlib'
        )
        
        assert result.field_type == 'scalar'
        assert result.bounds['x'] == (-3, 3)
        plt.close(result.figure)
    
    def test_electromagnetic_field(self):
        """Test electromagnetic field visualization"""
        # Simple dipole field
        def em_dipole(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
            # Simplified E and B fields
            Ex, Ey, Ez = x/r**3, y/r**3, z/r**3
            Bx, By, Bz = -y/r**3, x/r**3, 0
            return [Ex, Ey, Ez, Bx, By, Bz]
        
        result = field_visualization_3d(
            em_dipole,
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            field_type='electromagnetic',
            resolution=8,
            backend='matplotlib'
        )
        
        assert result.field_type == 'electromagnetic'
        plt.close(result.figure)
    
    def test_array_input(self):
        """Test with pre-computed array data"""
        # Create a simple vector field array
        nx, ny, nz = 10, 10, 10
        field_data = np.zeros((nx, ny, nz, 3))
        
        # Circular flow in xy plane
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        field_data[..., 0] = -Y
        field_data[..., 1] = X
        field_data[..., 2] = 0
        
        result = field_visualization_3d(
            field_data,
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            field_type='vector',
            backend='matplotlib'
        )
        
        assert result.statistics['max_magnitude'] > 0
        plt.close(result.figure)
    
    def test_normalization(self):
        """Test field normalization"""
        # Strong magnitude variation
        def varying_field(x, y, z):
            magnitude = 1 + 10 * np.exp(-(x**2 + y**2))
            return [magnitude, 0, 0]
        
        result_norm = field_visualization_3d(
            varying_field,
            bounds=((-2, 2), (-2, 2), (-2, 2)),
            field_type='vector',
            resolution=10,
            normalize=True,
            backend='matplotlib'
        )
        
        result_no_norm = field_visualization_3d(
            varying_field,
            bounds=((-2, 2), (-2, 2), (-2, 2)),
            field_type='vector',
            resolution=10,
            normalize=False,
            backend='matplotlib'
        )
        
        # Normalized should have different statistics
        assert result_norm.statistics['max_magnitude'] != result_no_norm.statistics['max_magnitude']
        plt.close(result_norm.figure)
        plt.close(result_no_norm.figure)
    
    def test_log_scale(self):
        """Test logarithmic scale option"""
        # Exponentially varying field
        def exp_field(x, y, z):
            return np.exp(x + y + z)
        
        result = field_visualization_3d(
            exp_field,
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            field_type='scalar',
            resolution=10,
            log_scale=True,
            backend='matplotlib'
        )
        
        assert result.field_type == 'scalar'
        plt.close(result.figure)
    
    def test_plotly_backend(self):
        """Test plotly backend"""
        def simple_field(x, y, z):
            return [1, 0, 0]
        
        result = field_visualization_3d(
            simple_field,
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            field_type='vector',
            resolution=5,
            backend='plotly'
        )
        
        # Check it's a plotly figure
        assert hasattr(result.figure, 'add_trace')
        assert hasattr(result.figure, 'update_layout')
    
    def test_resolution_options(self):
        """Test different resolution specifications"""
        def test_field(x, y, z):
            return [x, y, z]
        
        # Uniform resolution
        result1 = field_visualization_3d(
            test_field,
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            resolution=8,
            backend='matplotlib'
        )
        plt.close(result1.figure)
        
        # Non-uniform resolution
        result2 = field_visualization_3d(
            test_field,
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            resolution=(10, 8, 6),
            backend='matplotlib'
        )
        plt.close(result2.figure)
        
        assert result1.statistics is not None
        assert result2.statistics is not None
    
    def test_error_handling(self):
        """Test error handling"""
        def dummy_field(x, y, z):
            return [0, 0, 0]
        
        # Invalid backend
        with pytest.raises(ValueError):
            field_visualization_3d(
                dummy_field,
                bounds=((-1, 1), (-1, 1), (-1, 1)),
                backend='invalid'
            )
    
    def test_custom_colormap_and_alpha(self):
        """Test custom visualization parameters"""
        def test_field(x, y, z):
            return [np.sin(x), np.cos(y), np.sin(z)]
        
        result = field_visualization_3d(
            test_field,
            bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
            field_type='vector',
            resolution=8,
            colormap='plasma',
            alpha=0.5,
            backend='matplotlib'
        )
        
        assert isinstance(result.figure, plt.Figure)
        plt.close(result.figure)