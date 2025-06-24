"""
Tests for geometry and mesh support.
"""

import pytest
import torch
import numpy as np
from phizzyML.geometry.mesh import Mesh, MeshGenerator, AdaptiveMesh
from phizzyML.geometry.operators import LaplacianOperator, GradientOperator, DivergenceOperator


class TestMesh:
    def test_mesh_creation(self):
        """Test basic mesh creation."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
        
        mesh = Mesh(vertices, elements)
        
        assert mesh.n_vertices == 3
        assert mesh.n_elements == 1
        assert mesh.dimension == 2
    
    def test_triangle_area_computation(self):
        """Test triangle area calculation."""
        # Right triangle with legs of length 1
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
        
        mesh = Mesh(vertices, elements)
        areas = mesh.compute_element_volumes()
        
        # Area should be 0.5
        assert torch.allclose(areas, torch.tensor([0.5]), atol=1e-6)
    
    def test_edge_lengths(self):
        """Test edge length computation."""
        vertices = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
        
        mesh = Mesh(vertices, elements)
        edges = mesh.compute_edge_lengths()
        
        # Should have edges of length 3, 4, 5
        expected = torch.tensor([[3.0, 4.0, 5.0]])
        assert torch.allclose(edges, expected, atol=1e-6)
    
    def test_boundary_detection(self):
        """Test boundary vertex detection."""
        # Simple 2x2 grid
        mesh = MeshGenerator.rectangular_grid(3, 3)
        boundary_vertices = mesh.get_boundary_vertices()
        
        # Should have 8 boundary vertices (perimeter of 3x3 grid)
        assert len(boundary_vertices) == 8


class TestMeshGenerator:
    def test_rectangular_grid(self):
        """Test rectangular grid generation."""
        mesh = MeshGenerator.rectangular_grid(3, 3, (0, 2), (0, 2))
        
        # Should have 9 vertices
        assert mesh.n_vertices == 9
        
        # Should have proper bounds
        assert mesh.vertices[:, 0].min() == pytest.approx(0.0)
        assert mesh.vertices[:, 0].max() == pytest.approx(2.0)
        assert mesh.vertices[:, 1].min() == pytest.approx(0.0)
        assert mesh.vertices[:, 1].max() == pytest.approx(2.0)
    
    def test_circular_mesh(self):
        """Test circular mesh generation."""
        mesh = MeshGenerator.circular_mesh((0, 0), 1.0, n_radial=3, n_angular=6)
        
        # Check that outer vertices are on circle
        outer_vertices = mesh.vertices[mesh.boundary_markers > 0]
        distances = torch.norm(outer_vertices, dim=1)
        
        assert torch.allclose(distances, torch.ones_like(distances), atol=1e-6)
    
    def test_geometry_function_mesh(self):
        """Test mesh generation from geometry function."""
        # Circle of radius 0.5
        def circle(x, y):
            return x**2 + y**2 <= 0.25
        
        mesh = MeshGenerator.from_geometry_function(
            circle, (-1, 1), (-1, 1), resolution=0.2
        )
        
        # All vertices should be inside circle
        distances = torch.norm(mesh.vertices, dim=1)
        assert torch.all(distances <= 0.5 + 1e-6)


class TestAdaptiveMesh:
    def test_refinement_by_error(self):
        """Test adaptive mesh refinement."""
        # Start with coarse mesh
        mesh = MeshGenerator.rectangular_grid(3, 3)
        adaptive = AdaptiveMesh(mesh)
        
        # Create error field (high error in center)
        error_field = torch.zeros(mesh.n_vertices)
        center_idx = mesh.n_vertices // 2
        error_field[center_idx] = 10.0
        
        # Refine
        refined_mesh = adaptive.refine_by_error(error_field, threshold=5.0)
        
        # Should have more elements after refinement
        assert refined_mesh.n_elements >= mesh.n_elements


class TestDifferentialOperators:
    def test_laplacian_operator(self):
        """Test Laplacian operator construction."""
        mesh = MeshGenerator.rectangular_grid(5, 5)
        laplacian = LaplacianOperator(mesh)
        
        # Test on linear function (should give zero)
        x_coords = mesh.vertices[:, 0]
        result = laplacian(x_coords)
        
        # Laplacian of linear function should be close to zero (except boundaries)
        interior_mask = mesh.boundary_markers == 0
        assert torch.abs(result[interior_mask]).max() < 0.1
    
    def test_gradient_operator(self):
        """Test gradient operator."""
        mesh = MeshGenerator.rectangular_grid(4, 4)
        gradient = GradientOperator(mesh)
        
        # Test on quadratic function: f = x²
        x_coords = mesh.vertices[:, 0]
        quadratic_field = x_coords**2
        
        grad_result = gradient(quadratic_field)
        
        # Gradient should be approximately 2x in x-direction
        assert grad_result.shape[1] == 2  # 2D gradient
    
    def test_divergence_operator(self):
        """Test divergence operator."""
        mesh = MeshGenerator.rectangular_grid(4, 4)
        divergence = DivergenceOperator(mesh)
        
        # Create constant vector field
        n_elements = mesh.n_elements
        vector_field = torch.ones(n_elements, 2)
        
        div_result = divergence(vector_field)
        
        # Divergence of constant field should be zero
        assert div_result.shape[0] == mesh.n_vertices