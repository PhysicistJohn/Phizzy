"""
Differential operators on meshes for physics simulations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import scipy.sparse
from .mesh import Mesh


class DifferentialOperator(nn.Module):
    """Base class for differential operators on meshes."""
    
    def __init__(self, mesh: Mesh):
        super().__init__()
        self.mesh = mesh
        self._build_operator()
    
    def _build_operator(self):
        """Build the operator matrix. To be implemented by subclasses."""
        raise NotImplementedError
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Apply operator to field values."""
        raise NotImplementedError


class LaplacianOperator(DifferentialOperator):
    """
    Discrete Laplacian operator on mesh.
    Uses finite element method for general meshes.
    """
    
    def __init__(self, mesh: Mesh, boundary_type: str = 'dirichlet'):
        self.boundary_type = boundary_type
        super().__init__(mesh)
    
    def _build_operator(self):
        """Build discrete Laplacian matrix."""
        n = self.mesh.n_vertices
        
        # Initialize sparse matrix components
        row_indices = []
        col_indices = []
        values = []
        
        # For each element, compute local contributions
        for elem in self.mesh.elements:
            # Get element vertices
            vertices = self.mesh.vertices[elem]
            
            # Compute local stiffness matrix
            local_matrix = self._compute_element_laplacian(vertices)
            
            # Add to global matrix
            for i in range(len(elem)):
                for j in range(len(elem)):
                    row_indices.append(elem[i].item())
                    col_indices.append(elem[j].item())
                    values.append(local_matrix[i, j].item())
        
        # Convert to sparse tensor
        indices = torch.tensor([row_indices, col_indices])
        values = torch.tensor(values)
        self.laplacian_matrix = torch.sparse_coo_tensor(
            indices, values, (n, n), dtype=torch.float32
        )
        
        # Apply boundary conditions
        if self.boundary_type == 'dirichlet':
            self._apply_dirichlet_bc()
    
    def _compute_element_laplacian(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute local Laplacian matrix for an element.
        Uses linear finite elements.
        """
        if vertices.shape[0] == 3 and vertices.shape[1] == 2:
            # 2D triangle
            return self._laplacian_triangle_2d(vertices)
        elif vertices.shape[0] == 4 and vertices.shape[1] == 3:
            # 3D tetrahedron
            return self._laplacian_tetrahedron_3d(vertices)
        else:
            raise NotImplementedError(f"Element type not supported")
    
    def _laplacian_triangle_2d(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian for 2D triangular element."""
        # Vertices: v0, v1, v2
        v0, v1, v2 = vertices
        
        # Compute area
        area = 0.5 * torch.abs(
            (v1[0] - v0[0]) * (v2[1] - v0[1]) - 
            (v1[1] - v0[1]) * (v2[0] - v0[0])
        )
        
        # Shape function gradients (constant for linear elements)
        # grad_phi_i = 1/(2*area) * (perpendicular to opposite edge)
        grad_phi = torch.zeros(3, 2)
        grad_phi[0] = torch.tensor([v2[1] - v1[1], v1[0] - v2[0]]) / (2 * area)
        grad_phi[1] = torch.tensor([v0[1] - v2[1], v2[0] - v0[0]]) / (2 * area)
        grad_phi[2] = torch.tensor([v1[1] - v0[1], v0[0] - v1[0]]) / (2 * area)
        
        # Local stiffness matrix: K_ij = area * dot(grad_phi_i, grad_phi_j)
        local_matrix = area * torch.matmul(grad_phi, grad_phi.T)
        
        return local_matrix
    
    def _laplacian_tetrahedron_3d(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian for 3D tetrahedral element."""
        # Similar to 2D but in 3D
        # This is a simplified implementation
        n = vertices.shape[0]
        local_matrix = torch.zeros(n, n)
        
        # Compute volume
        v0, v1, v2, v3 = vertices
        mat = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=1)
        volume = torch.abs(torch.det(mat)) / 6.0
        
        # Simplified: use approximate gradients
        # Real implementation would compute proper shape function gradients
        
        return local_matrix
    
    def _apply_dirichlet_bc(self):
        """Apply Dirichlet boundary conditions."""
        if self.mesh.boundary_markers is not None:
            boundary_indices = torch.where(self.mesh.boundary_markers > 0)[0]
            
            # Zero out boundary rows/columns
            # This is simplified - proper implementation would modify matrix
            pass
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian operator to field."""
        # Ensure float type to match laplacian_matrix
        field_float = field.float()
        
        if field_float.dim() == 1:
            # Direct matrix-vector multiplication
            return torch.sparse.mm(self.laplacian_matrix, field_float.unsqueeze(1)).squeeze()
        else:
            # Apply to each component
            result = []
            for i in range(field_float.shape[1]):
                result.append(
                    torch.sparse.mm(self.laplacian_matrix, field_float[:, i].unsqueeze(1)).squeeze()
                )
            return torch.stack(result, dim=1)


class GradientOperator(DifferentialOperator):
    """
    Discrete gradient operator on mesh.
    Computes gradient of scalar field as vector field.
    """
    
    def _build_operator(self):
        """Build gradient operator matrices."""
        # For each spatial dimension, build a gradient matrix
        self.grad_matrices = []
        
        for dim in range(self.mesh.dimension):
            row_indices = []
            col_indices = []
            values = []
            
            # Build gradient in each direction
            for elem_idx, elem in enumerate(self.mesh.elements):
                vertices = self.mesh.vertices[elem]
                grad_weights = self._compute_element_gradients(vertices, dim)
                
                for i, vertex_idx in enumerate(elem):
                    # Each element contributes to its vertices
                    row_indices.append(elem_idx)
                    col_indices.append(vertex_idx.item())
                    values.append(grad_weights[i].item())
            
            # Create sparse matrix
            n_elements = self.mesh.n_elements
            n_vertices = self.mesh.n_vertices
            indices = torch.tensor([row_indices, col_indices])
            values = torch.tensor(values)
            
            grad_matrix = torch.sparse_coo_tensor(
                indices, values, (n_elements, n_vertices), dtype=torch.float32
            )
            self.grad_matrices.append(grad_matrix)
    
    def _compute_element_gradients(self, vertices: torch.Tensor, 
                                  dim: int) -> torch.Tensor:
        """Compute gradient weights for element in given dimension."""
        if vertices.shape[0] == 3 and vertices.shape[1] == 2:
            # 2D triangle
            v0, v1, v2 = vertices
            area = 0.5 * torch.abs(
                (v1[0] - v0[0]) * (v2[1] - v0[1]) - 
                (v1[1] - v0[1]) * (v2[0] - v0[0])
            )
            
            if dim == 0:  # x-gradient
                weights = torch.tensor([v2[1] - v1[1], v0[1] - v2[1], v1[1] - v0[1]]) / (2 * area)
            else:  # y-gradient
                weights = torch.tensor([v1[0] - v2[0], v2[0] - v0[0], v0[0] - v1[0]]) / (2 * area)
            
            return weights / 3.0  # Average over element
        else:
            # Placeholder for other element types
            return torch.zeros(vertices.shape[0])
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of scalar field.
        Returns vector field on elements.
        """
        gradients = []
        
        for grad_matrix in self.grad_matrices:
            grad_component = torch.sparse.mm(grad_matrix, field.unsqueeze(1)).squeeze()
            gradients.append(grad_component)
        
        return torch.stack(gradients, dim=1)


class DivergenceOperator(DifferentialOperator):
    """
    Discrete divergence operator on mesh.
    Computes divergence of vector field.
    """
    
    def __init__(self, mesh: Mesh):
        super().__init__(mesh)
        # Divergence is negative transpose of gradient
        self.gradient_op = GradientOperator(mesh)
    
    def _build_operator(self):
        """Divergence operator is built from gradient operator."""
        pass
    
    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of vector field.
        Input: vector field on elements (n_elements, dim)
        Output: scalar field on vertices (n_vertices,)
        """
        divergence = torch.zeros(self.mesh.n_vertices)
        
        # For each dimension, accumulate contributions
        for dim in range(self.mesh.dimension):
            # Use transpose of gradient matrix
            grad_matrix = self.gradient_op.grad_matrices[dim]
            
            # Negative transpose for divergence
            div_contribution = -torch.sparse.mm(
                grad_matrix.t(), 
                vector_field[:, dim].unsqueeze(1)
            ).squeeze()
            
            divergence += div_contribution
        
        return divergence


class PhysicsInformedLoss(nn.Module):
    """
    Loss function that incorporates differential operators.
    Useful for physics-informed neural networks on meshes.
    """
    
    def __init__(self, mesh: Mesh, equation_type: str = 'poisson'):
        super().__init__()
        self.mesh = mesh
        self.equation_type = equation_type
        
        # Initialize operators
        self.laplacian = LaplacianOperator(mesh)
        self.gradient = GradientOperator(mesh)
        self.divergence = DivergenceOperator(mesh)
    
    def forward(self, predicted: torch.Tensor, source: torch.Tensor,
                boundary_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            predicted: Predicted field values on mesh vertices
            source: Source term values
            boundary_values: Known boundary values (optional)
        
        Returns:
            Total loss combining PDE residual and boundary conditions
        """
        if self.equation_type == 'poisson':
            # Poisson equation: -∇²u = f
            laplacian_u = self.laplacian(predicted)
            pde_residual = laplacian_u + source
            pde_loss = torch.mean(pde_residual**2)
        
        elif self.equation_type == 'heat':
            # Heat equation: ∂u/∂t - α∇²u = 0
            # This would need time derivative handling
            laplacian_u = self.laplacian(predicted)
            pde_loss = torch.mean(laplacian_u**2)
        
        else:
            raise ValueError(f"Unknown equation type: {self.equation_type}")
        
        # Boundary condition loss
        bc_loss = 0.0
        if boundary_values is not None and self.mesh.boundary_markers is not None:
            boundary_indices = torch.where(self.mesh.boundary_markers > 0)[0]
            bc_residual = predicted[boundary_indices] - boundary_values
            bc_loss = torch.mean(bc_residual**2)
        
        return pde_loss + 10.0 * bc_loss  # Weight boundary conditions higher