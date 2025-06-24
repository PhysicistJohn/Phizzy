"""
Mesh and geometry support for physics simulations.
Handles complex geometries and adaptive mesh refinement.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
import scipy.spatial


@dataclass
class Mesh:
    """
    General mesh representation for physics simulations.
    """
    vertices: torch.Tensor  # (N, D) vertex coordinates
    elements: torch.Tensor  # (M, K) element connectivity
    boundary_markers: Optional[torch.Tensor] = None  # Boundary identification
    vertex_data: Optional[Dict[str, torch.Tensor]] = None  # Data on vertices
    element_data: Optional[Dict[str, torch.Tensor]] = None  # Data on elements
    
    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]
    
    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]
    
    @property
    def dimension(self) -> int:
        return self.vertices.shape[1]
    
    def compute_element_volumes(self) -> torch.Tensor:
        """Compute volume/area of each element."""
        if self.dimension == 2 and self.elements.shape[1] == 3:
            # 2D triangular mesh
            return self._compute_triangle_areas()
        elif self.dimension == 3 and self.elements.shape[1] == 4:
            # 3D tetrahedral mesh
            return self._compute_tetrahedron_volumes()
        else:
            raise NotImplementedError(f"Volume computation not implemented for this mesh type")
    
    def _compute_triangle_areas(self) -> torch.Tensor:
        """Compute areas of triangular elements."""
        v0 = self.vertices[self.elements[:, 0]]
        v1 = self.vertices[self.elements[:, 1]]
        v2 = self.vertices[self.elements[:, 2]]
        
        # Cross product for area
        areas = 0.5 * torch.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        )
        return areas
    
    def _compute_tetrahedron_volumes(self) -> torch.Tensor:
        """Compute volumes of tetrahedral elements."""
        v0 = self.vertices[self.elements[:, 0]]
        v1 = self.vertices[self.elements[:, 1]]
        v2 = self.vertices[self.elements[:, 2]]
        v3 = self.vertices[self.elements[:, 3]]
        
        # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
        mat = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)
        volumes = torch.abs(torch.det(mat)) / 6.0
        return volumes
    
    def compute_edge_lengths(self) -> torch.Tensor:
        """Compute all edge lengths in the mesh."""
        edge_lengths = []
        n_edges_per_element = self.elements.shape[1]
        
        for i in range(n_edges_per_element):
            for j in range(i + 1, n_edges_per_element):
                v_i = self.vertices[self.elements[:, i]]
                v_j = self.vertices[self.elements[:, j]]
                lengths = torch.norm(v_j - v_i, dim=1)
                edge_lengths.append(lengths)
        
        return torch.stack(edge_lengths, dim=1)
    
    def get_boundary_vertices(self) -> torch.Tensor:
        """Get indices of boundary vertices."""
        if self.boundary_markers is not None:
            return torch.where(self.boundary_markers > 0)[0]
        else:
            # Simple heuristic: vertices that appear in fewer elements
            vertex_count = torch.zeros(self.n_vertices)
            for elem in self.elements:
                vertex_count[elem] += 1
            
            threshold = torch.median(vertex_count) * 0.8
            return torch.where(vertex_count < threshold)[0]
    
    def interpolate_to_points(self, points: torch.Tensor, 
                            field: torch.Tensor) -> torch.Tensor:
        """
        Interpolate field values from mesh to arbitrary points.
        Uses barycentric interpolation for simplices.
        """
        # Find which element each point belongs to
        # This is a simplified implementation - real implementation would use
        # spatial data structures for efficiency
        
        interpolated = torch.zeros(points.shape[0], device=points.device)
        
        for i, point in enumerate(points):
            # Find containing element (simplified)
            # In practice, use spatial search structures
            min_dist = float('inf')
            closest_vertex = 0
            
            for j, vertex in enumerate(self.vertices):
                dist = torch.norm(point - vertex)
                if dist < min_dist:
                    min_dist = dist
                    closest_vertex = j
            
            interpolated[i] = field[closest_vertex]
        
        return interpolated


class MeshGenerator:
    """Generate common mesh types for physics simulations."""
    
    @staticmethod
    def rectangular_grid(nx: int, ny: int, 
                        x_range: Tuple[float, float] = (0, 1),
                        y_range: Tuple[float, float] = (0, 1)) -> Mesh:
        """Generate a rectangular grid mesh."""
        x = torch.linspace(x_range[0], x_range[1], nx)
        y = torch.linspace(y_range[0], y_range[1], ny)
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        vertices = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Create triangular elements
        elements = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Bottom-left vertex index
                idx = i * ny + j
                
                # First triangle
                elements.append([idx, idx + 1, idx + ny])
                # Second triangle
                elements.append([idx + 1, idx + ny + 1, idx + ny])
        
        elements = torch.tensor(elements, dtype=torch.long)
        
        # Mark boundary vertices
        boundary_markers = torch.zeros(vertices.shape[0])
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_markers[idx] = 1
        
        return Mesh(vertices, elements, boundary_markers)
    
    @staticmethod
    def circular_mesh(center: Tuple[float, float], radius: float,
                     n_radial: int = 10, n_angular: int = 20) -> Mesh:
        """Generate a circular mesh."""
        vertices = [torch.tensor(center)]  # Center point
        
        # Generate vertices in rings
        for r_idx in range(1, n_radial + 1):
            r = radius * r_idx / n_radial
            for a_idx in range(n_angular):
                angle = 2 * np.pi * a_idx / n_angular
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                vertices.append(torch.tensor([x, y]))
        
        vertices = torch.stack(vertices)
        
        # Create triangular elements
        elements = []
        
        # Center triangles
        for a_idx in range(n_angular):
            next_a = (a_idx + 1) % n_angular
            elements.append([0, 1 + a_idx, 1 + next_a])
        
        # Ring triangles
        for r_idx in range(n_radial - 1):
            ring_start = 1 + r_idx * n_angular
            next_ring_start = ring_start + n_angular
            
            for a_idx in range(n_angular):
                next_a = (a_idx + 1) % n_angular
                
                # Two triangles per quad
                v1 = ring_start + a_idx
                v2 = ring_start + next_a
                v3 = next_ring_start + a_idx
                v4 = next_ring_start + next_a
                
                elements.append([v1, v2, v3])
                elements.append([v2, v4, v3])
        
        elements = torch.tensor(elements, dtype=torch.long)
        
        # Mark boundary
        boundary_markers = torch.zeros(vertices.shape[0])
        outer_ring_start = 1 + (n_radial - 1) * n_angular
        boundary_markers[outer_ring_start:] = 1
        
        return Mesh(vertices, elements, boundary_markers)
    
    @staticmethod
    def from_geometry_function(func: Callable[[float, float], bool],
                             x_range: Tuple[float, float],
                             y_range: Tuple[float, float],
                             resolution: float) -> Mesh:
        """
        Generate mesh for arbitrary 2D geometry defined by a function.
        func(x, y) returns True if point is inside the geometry.
        """
        # Create initial grid
        nx = int((x_range[1] - x_range[0]) / resolution)
        ny = int((y_range[1] - y_range[0]) / resolution)
        
        mesh = MeshGenerator.rectangular_grid(nx, ny, x_range, y_range)
        
        # Filter vertices inside geometry
        inside = torch.tensor([
            func(v[0].item(), v[1].item()) for v in mesh.vertices
        ])
        
        # Remove elements with vertices outside
        valid_elements = []
        for elem in mesh.elements:
            if all(inside[elem]):
                valid_elements.append(elem)
        
        if not valid_elements:
            raise ValueError("No elements inside geometry")
        
        valid_elements = torch.stack(valid_elements)
        
        # Reindex vertices
        used_vertices = torch.unique(valid_elements.flatten())
        new_indices = torch.zeros(mesh.n_vertices, dtype=torch.long) - 1
        new_indices[used_vertices] = torch.arange(len(used_vertices))
        
        new_vertices = mesh.vertices[used_vertices]
        new_elements = new_indices[valid_elements]
        
        return Mesh(new_vertices, new_elements)


class AdaptiveMesh:
    """
    Adaptive mesh refinement for physics simulations.
    """
    
    def __init__(self, initial_mesh: Mesh):
        self.mesh = initial_mesh
        self.refinement_history = []
    
    def refine_by_error(self, error_field: torch.Tensor, 
                       threshold: float) -> Mesh:
        """
        Refine mesh elements where error exceeds threshold.
        """
        # Compute error per element
        element_errors = self._compute_element_average(error_field)
        
        # Mark elements for refinement
        refine_mask = element_errors > threshold
        
        if not refine_mask.any():
            return self.mesh
        
        # Refine marked elements
        new_vertices = list(self.mesh.vertices)
        new_elements = []
        
        for elem_idx, elem in enumerate(self.mesh.elements):
            if refine_mask[elem_idx]:
                # Refine this element by splitting edges
                new_elems, new_verts = self._refine_element(elem, new_vertices)
                new_elements.extend(new_elems)
                new_vertices.extend(new_verts)
            else:
                # Keep original element
                new_elements.append(elem)
        
        # Create new mesh
        new_vertices = torch.stack(new_vertices)
        new_elements = torch.tensor(new_elements, dtype=torch.long)
        
        refined_mesh = Mesh(new_vertices, new_elements)
        self.refinement_history.append(self.mesh)
        self.mesh = refined_mesh
        
        return refined_mesh
    
    def _compute_element_average(self, vertex_field: torch.Tensor) -> torch.Tensor:
        """Compute average field value per element."""
        element_values = []
        
        for elem in self.mesh.elements:
            elem_avg = vertex_field[elem].mean()
            element_values.append(elem_avg)
        
        return torch.stack(element_values)
    
    def _refine_element(self, element: torch.Tensor, 
                       vertices: List[torch.Tensor]) -> Tuple[List, List]:
        """
        Refine a single element by edge subdivision.
        Returns new elements and new vertices.
        """
        if element.shape[0] == 3:  # Triangle
            # Get vertices
            v0, v1, v2 = element
            
            # Create edge midpoints
            m01 = (vertices[v0] + vertices[v1]) / 2
            m12 = (vertices[v1] + vertices[v2]) / 2
            m20 = (vertices[v2] + vertices[v0]) / 2
            
            # Add new vertices
            n = len(vertices)
            new_vertices = [m01, m12, m20]
            
            # Create 4 new triangles
            new_elements = [
                torch.tensor([v0, n, n+2]),      # Corner 0
                torch.tensor([v1, n+1, n]),      # Corner 1
                torch.tensor([v2, n+2, n+1]),    # Corner 2
                torch.tensor([n, n+1, n+2])      # Center
            ]
            
            return new_elements, new_vertices
        else:
            # For other element types, just return original
            return [element], []
    
    def coarsen_by_gradient(self, field: torch.Tensor, 
                          min_gradient: float) -> Mesh:
        """
        Coarsen mesh in regions with low field gradients.
        """
        # Compute gradients
        gradients = self._compute_vertex_gradients(field)
        gradient_magnitudes = torch.norm(gradients, dim=1)
        
        # This is a simplified implementation
        # Real coarsening would merge elements carefully
        print("Mesh coarsening not fully implemented")
        return self.mesh
    
    def _compute_vertex_gradients(self, field: torch.Tensor) -> torch.Tensor:
        """Compute gradients at vertices using least squares."""
        gradients = torch.zeros(self.mesh.n_vertices, self.mesh.dimension)
        
        # Simplified: use finite differences with neighbors
        # Real implementation would use proper FEM gradient recovery
        
        return gradients