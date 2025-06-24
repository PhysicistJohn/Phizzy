import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Union, Callable, Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import warnings


@dataclass
class FieldVisualizationResult:
    """Result of 3D field visualization"""
    figure: Union[plt.Figure, go.Figure]
    field_type: str
    bounds: Dict[str, Tuple[float, float]]
    statistics: Dict[str, Any]
    field_data: Optional[np.ndarray] = None
    streamlines: Optional[np.ndarray] = None
    isosurfaces: Optional[List[np.ndarray]] = None
    
    def save(self, filename: str):
        """Save visualization to file"""
        if isinstance(self.figure, plt.Figure):
            self.figure.savefig(filename)
        else:  # Plotly figure
            self.figure.write_html(filename.replace('.png', '.html'))
            self.figure.write_image(filename)
    
    def show(self):
        """Display the visualization"""
        if isinstance(self.figure, plt.Figure):
            plt.show()
        else:
            self.figure.show()


def field_visualization_3d(
    field_func: Union[Callable, np.ndarray],
    bounds: Union[Tuple[Tuple[float, float], ...], Dict[str, Tuple[float, float]]],
    field_type: str = 'vector',
    resolution: Union[int, Tuple[int, int, int]] = 20,
    colormap: str = 'viridis',
    backend: str = 'matplotlib',
    streamlines: bool = True,
    arrows: bool = True,
    isosurfaces: Union[bool, List[float]] = False,
    slice_planes: Optional[List[str]] = None,
    alpha: float = 0.7,
    normalize: bool = True,
    log_scale: bool = False,
    interactive: bool = True,
    title: Optional[str] = None
) -> FieldVisualizationResult:
    """
    Intelligent 3D visualization of vector and tensor fields with automatic scaling.
    
    Parameters
    ----------
    field_func : callable or array
        Either a function f(x, y, z) returning field values, or a 4D array [nx, ny, nz, components]
    bounds : tuple or dict
        Spatial bounds as ((xmin, xmax), (ymin, ymax), (zmin, zmax)) or dict
    field_type : str
        Type of field: 'vector', 'scalar', 'tensor', 'electromagnetic'
    resolution : int or tuple
        Grid resolution (uniform or per-axis)
    colormap : str
        Matplotlib colormap name
    backend : str
        Visualization backend: 'matplotlib' or 'plotly'
    streamlines : bool
        Draw streamlines for vector fields
    arrows : bool
        Draw arrows for vector fields
    isosurfaces : bool or list
        Draw isosurfaces at specified values
    slice_planes : list
        List of slice planes: ['xy', 'xz', 'yz']
    alpha : float
        Transparency for 3D elements
    normalize : bool
        Normalize field magnitudes
    log_scale : bool
        Use logarithmic scale for magnitudes
    interactive : bool
        Enable interactive features (for plotly)
    title : str
        Plot title
        
    Returns
    -------
    FieldVisualizationResult
        Visualization result with figure and metadata
    """
    
    # Parse bounds
    if isinstance(bounds, dict):
        x_bounds = bounds.get('x', (-1, 1))
        y_bounds = bounds.get('y', (-1, 1))
        z_bounds = bounds.get('z', (-1, 1))
    else:
        x_bounds, y_bounds, z_bounds = bounds
    
    # Parse resolution
    if isinstance(resolution, int):
        nx = ny = nz = resolution
    else:
        nx, ny, nz = resolution
    
    # Create grid
    x = np.linspace(x_bounds[0], x_bounds[1], nx)
    y = np.linspace(y_bounds[0], y_bounds[1], ny)
    z = np.linspace(z_bounds[0], z_bounds[1], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Evaluate field
    if callable(field_func):
        if field_type == 'vector':
            # Expect function to return (u, v, w) components
            field_data = np.zeros((nx, ny, nz, 3))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        result = field_func(X[i,j,k], Y[i,j,k], Z[i,j,k])
                        if isinstance(result, (list, tuple, np.ndarray)):
                            field_data[i,j,k,:] = result[:3]
                        else:
                            field_data[i,j,k,0] = result
        elif field_type == 'scalar':
            field_data = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        field_data[i,j,k] = field_func(X[i,j,k], Y[i,j,k], Z[i,j,k])
        elif field_type == 'electromagnetic':
            # Expect function to return (Ex, Ey, Ez, Bx, By, Bz)
            field_data = np.zeros((nx, ny, nz, 6))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        field_data[i,j,k,:] = field_func(X[i,j,k], Y[i,j,k], Z[i,j,k])
    else:
        field_data = field_func
    
    # Compute statistics
    if field_type == 'vector':
        magnitude = np.sqrt(np.sum(field_data[...,:3]**2, axis=-1))
    elif field_type == 'scalar':
        magnitude = np.abs(field_data)
    elif field_type == 'electromagnetic':
        e_mag = np.sqrt(np.sum(field_data[...,:3]**2, axis=-1))
        b_mag = np.sqrt(np.sum(field_data[...,3:6]**2, axis=-1))
        magnitude = np.sqrt(e_mag**2 + b_mag**2)
    
    statistics = {
        'max_magnitude': np.max(magnitude),
        'min_magnitude': np.min(magnitude),
        'mean_magnitude': np.mean(magnitude),
        'std_magnitude': np.std(magnitude)
    }
    
    # Apply transformations
    if normalize and field_type == 'vector':
        # Normalize vectors while preserving zero vectors
        nonzero = magnitude > 1e-10
        field_data[nonzero, :3] /= magnitude[nonzero, np.newaxis]
    
    if log_scale:
        magnitude = np.log10(magnitude + 1e-10)
    
    # Create visualization
    if backend == 'matplotlib':
        fig = _create_matplotlib_visualization(
            X, Y, Z, field_data, magnitude, field_type,
            colormap, streamlines, arrows, isosurfaces,
            slice_planes, alpha, title
        )
    elif backend == 'plotly':
        fig = _create_plotly_visualization(
            X, Y, Z, field_data, magnitude, field_type,
            colormap, streamlines, arrows, isosurfaces,
            slice_planes, alpha, interactive, title
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return FieldVisualizationResult(
        figure=fig,
        field_type=field_type,
        bounds={'x': x_bounds, 'y': y_bounds, 'z': z_bounds},
        statistics=statistics,
        field_data=field_data
    )


def _create_matplotlib_visualization(X, Y, Z, field_data, magnitude, field_type,
                                   colormap, streamlines, arrows, isosurfaces,
                                   slice_planes, alpha, title):
    """Create matplotlib 3D visualization"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if field_type == 'vector':
        U, V, W = field_data[..., 0], field_data[..., 1], field_data[..., 2]
        
        if arrows:
            # Subsample for arrow plot
            skip = max(1, min(X.shape) // 10)
            ax.quiver(X[::skip, ::skip, ::skip],
                     Y[::skip, ::skip, ::skip],
                     Z[::skip, ::skip, ::skip],
                     U[::skip, ::skip, ::skip],
                     V[::skip, ::skip, ::skip],
                     W[::skip, ::skip, ::skip],
                     length=0.1, normalize=True, alpha=alpha)
        
        if streamlines:
            # Create streamlines (simplified - just show a few seed points)
            seeds = np.random.rand(10, 3)
            for seed in seeds:
                start_point = [
                    X.min() + seed[0] * (X.max() - X.min()),
                    Y.min() + seed[1] * (Y.max() - Y.min()),
                    Z.min() + seed[2] * (Z.max() - Z.min())
                ]
                # Would integrate streamline here
                pass
    
    elif field_type == 'scalar':
        # Show isosurfaces
        if isinstance(isosurfaces, list):
            for iso_val in isosurfaces:
                # Would use marching cubes here
                pass
        
        # Show slice planes
        if slice_planes:
            if 'xy' in slice_planes:
                z_idx = Z.shape[2] // 2
                c = ax.contourf(X[:, :, z_idx], Y[:, :, z_idx], 
                               field_data[:, :, z_idx], 
                               cmap=colormap, alpha=alpha)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    
    return fig


def _create_plotly_visualization(X, Y, Z, field_data, magnitude, field_type,
                               colormap, streamlines, arrows, isosurfaces,
                               slice_planes, alpha, interactive, title):
    """Create plotly 3D visualization"""
    fig = go.Figure()
    
    if field_type == 'vector':
        if arrows:
            # Create cone plot for vector field
            skip = max(1, min(X.shape) // 15)
            x_sub = X[::skip, ::skip, ::skip].flatten()
            y_sub = Y[::skip, ::skip, ::skip].flatten()
            z_sub = Z[::skip, ::skip, ::skip].flatten()
            u_sub = field_data[::skip, ::skip, ::skip, 0].flatten()
            v_sub = field_data[::skip, ::skip, ::skip, 1].flatten()
            w_sub = field_data[::skip, ::skip, ::skip, 2].flatten()
            
            fig.add_trace(go.Cone(
                x=x_sub, y=y_sub, z=z_sub,
                u=u_sub, v=v_sub, w=w_sub,
                colorscale=colormap,
                opacity=alpha,
                sizemode='scaled',
                sizeref=0.5
            ))
    
    elif field_type == 'scalar':
        if isinstance(isosurfaces, list) and len(isosurfaces) > 0:
            # Create isosurfaces
            for iso_val in isosurfaces:
                fig.add_trace(go.Isosurface(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                    value=field_data.flatten(),
                    isomin=iso_val - 0.01,
                    isomax=iso_val + 0.01,
                    surface_count=1,
                    opacity=alpha,
                    colorscale=colormap
                ))
    
    elif field_type == 'electromagnetic':
        # Show E and B fields with different colors
        skip = max(1, min(X.shape) // 12)
        
        # E field in red
        fig.add_trace(go.Cone(
            x=X[::skip, ::skip, ::skip].flatten(),
            y=Y[::skip, ::skip, ::skip].flatten(),
            z=Z[::skip, ::skip, ::skip].flatten(),
            u=field_data[::skip, ::skip, ::skip, 0].flatten(),
            v=field_data[::skip, ::skip, ::skip, 1].flatten(),
            w=field_data[::skip, ::skip, ::skip, 2].flatten(),
            colorscale='Reds',
            opacity=alpha,
            name='E field'
        ))
        
        # B field in blue
        fig.add_trace(go.Cone(
            x=X[::skip, ::skip, ::skip].flatten(),
            y=Y[::skip, ::skip, ::skip].flatten(),
            z=Z[::skip, ::skip, ::skip].flatten(),
            u=field_data[::skip, ::skip, ::skip, 3].flatten(),
            v=field_data[::skip, ::skip, ::skip, 4].flatten(),
            w=field_data[::skip, ::skip, ::skip, 5].flatten(),
            colorscale='Blues',
            opacity=alpha,
            name='B field'
        ))
    
    fig.update_layout(
        title=title or f'{field_type.capitalize()} Field Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        showlegend=True
    )
    
    return fig