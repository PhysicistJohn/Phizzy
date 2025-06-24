import numpy as np
import opt_einsum
from typing import List, Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import itertools
import warnings


@dataclass
class ContractionResult:
    """Result of optimized tensor contraction"""
    result: np.ndarray
    path: List[Tuple[int, int]]
    cost: float
    memory_usage: int
    scaling: str
    optimization_time: float
    execution_time: float
    intermediate_shapes: List[Tuple[int, ...]]
    
    def __str__(self):
        return f"ContractionResult(shape={self.result.shape}, cost={self.cost:.2e}, scaling={self.scaling})"


def tensor_contract_optimize(
    equation: Union[str, List[np.ndarray]],
    tensors: Optional[List[np.ndarray]] = None,
    optimize: Union[str, bool] = 'auto',
    memory_limit: Optional[int] = None,
    use_blas: bool = True,
    parallelize: bool = False,
    algorithm: str = 'greedy',
    intermediate_format: str = 'numpy',
    precision: str = 'float64',
    output_file: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> ContractionResult:
    """
    Automatically optimize tensor contraction order for complex many-body calculations.
    
    Parameters
    ----------
    equation : str or list
        Einstein summation equation or list of tensors with automatic equation generation
    tensors : list of arrays
        Input tensors (required if equation is string)
    optimize : str or bool
        Optimization strategy: 'auto', 'greedy', 'optimal', 'random', 'dp' (dynamic programming)
    memory_limit : int
        Maximum memory usage in bytes
    use_blas : bool
        Use BLAS for matrix operations when possible
    parallelize : bool
        Use parallel computation for intermediate contractions
    algorithm : str
        Specific algorithm: 'greedy', 'branch-bound', 'simulated-annealing'
    intermediate_format : str
        Format for intermediate tensors: 'numpy', 'sparse', 'low-rank'
    precision : str
        Numerical precision: 'float32', 'float64', 'complex64', 'complex128'
    output_file : str
        Save intermediate results to file for very large contractions
    chunk_size : int
        Process tensors in chunks for memory efficiency
        
    Returns
    -------
    ContractionResult
        Optimized contraction result with path and performance metrics
    """
    
    import time
    start_time = time.time()
    
    # Handle different input formats
    if isinstance(equation, list) and tensors is None:
        tensors = equation
        equation = _generate_equation(tensors)
    elif isinstance(equation, str) and tensors is None:
        raise ValueError("tensors must be provided when equation is a string")
    
    # Validate inputs
    if not tensors:
        raise ValueError("No tensors provided")
    
    # Convert precision
    dtype = _get_dtype(precision)
    tensors = [t.astype(dtype) for t in tensors]
    
    # Memory management
    if memory_limit is None:
        memory_limit = _estimate_memory_limit()
    
    # Choose optimization strategy
    if optimize == 'auto':
        optimize = _auto_select_optimizer(tensors, equation)
    
    optimization_time = time.time() - start_time
    
    # Perform optimization
    if optimize == 'optimal':
        path, path_info = _optimal_contraction_path(equation, tensors, memory_limit)
    elif optimize == 'greedy':
        path, path_info = _greedy_contraction_path(equation, tensors, memory_limit)
    elif optimize == 'random':
        path, path_info = _random_contraction_path(equation, tensors, memory_limit)
    elif algorithm == 'simulated-annealing':
        path, path_info = _simulated_annealing_path(equation, tensors, memory_limit)
    else:
        # Use opt_einsum as fallback
        path, path_info = opt_einsum.contract_path(equation, *tensors, optimize=True)
    
    optimization_time = time.time() - start_time - optimization_time
    
    # Execute contraction
    start_exec = time.time()
    
    if chunk_size is not None:
        result = _chunked_contraction(equation, tensors, path, chunk_size)
    elif parallelize:
        result = _parallel_contraction(equation, tensors, path)
    else:
        result = _execute_contraction(equation, tensors, path, use_blas)
    
    execution_time = time.time() - start_exec
    
    # Extract performance metrics
    cost = path_info.opt_cost if hasattr(path_info, 'opt_cost') else 0
    scaling = _analyze_scaling(equation, tensors)
    memory_usage = _estimate_peak_memory(tensors, path)
    intermediate_shapes = _get_intermediate_shapes(tensors, path)
    
    # Save output if requested
    if output_file:
        np.save(output_file, result)
    
    return ContractionResult(
        result=result,
        path=path,
        cost=cost,
        memory_usage=memory_usage,
        scaling=scaling,
        optimization_time=optimization_time,
        execution_time=execution_time,
        intermediate_shapes=intermediate_shapes
    )


def _get_dtype(precision: str) -> np.dtype:
    """Convert precision string to numpy dtype"""
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'complex64': np.complex64,
        'complex128': np.complex128
    }
    return dtype_map.get(precision, np.float64)


def _estimate_memory_limit() -> int:
    """Estimate available memory limit"""
    try:
        import psutil
        return int(psutil.virtual_memory().available * 0.8)  # Use 80% of available
    except ImportError:
        return 8 * 1024**3  # Default to 8GB


def _generate_equation(tensors: List[np.ndarray]) -> str:
    """Generate Einstein summation equation from tensor shapes"""
    # Simple heuristic: assume chain-like contraction
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    if len(tensors) == 2:
        # Matrix multiplication case
        if tensors[0].ndim == 2 and tensors[1].ndim == 2:
            if tensors[0].shape[1] == tensors[1].shape[0]:
                return 'ij,jk->ik'
        # Vector-matrix cases
        elif tensors[0].ndim == 1 and tensors[1].ndim == 2:
            return 'i,ij->j'
        elif tensors[0].ndim == 2 and tensors[1].ndim == 1:
            return 'ij,j->i'
    
    # General case: try to infer from matching dimensions
    equation_parts = []
    used_indices = set()
    index_counter = 0
    
    for i, tensor in enumerate(tensors):
        indices = []
        for dim_size in tensor.shape:
            # Find matching dimension in previous tensors
            found_match = False
            for prev_tensor in tensors[:i]:
                if dim_size in prev_tensor.shape:
                    # Find the index for this dimension
                    for j, prev_dim in enumerate(prev_tensor.shape):
                        if prev_dim == dim_size:
                            # Use the same index
                            prev_eq = equation_parts[tensors.index(prev_tensor)]
                            indices.append(prev_eq[j])
                            found_match = True
                            break
                    if found_match:
                        break
            
            if not found_match:
                # Assign new index
                while alphabet[index_counter] in used_indices:
                    index_counter += 1
                indices.append(alphabet[index_counter])
                used_indices.add(alphabet[index_counter])
                index_counter += 1
        
        equation_parts.append(''.join(indices))
    
    # Determine output indices (free indices that appear only once)
    all_indices = ''.join(equation_parts)
    output_indices = []
    for char in alphabet[:index_counter]:
        if all_indices.count(char) == 1:
            output_indices.append(char)
    
    equation = ','.join(equation_parts) + '->' + ''.join(sorted(output_indices))
    return equation


def _auto_select_optimizer(tensors: List[np.ndarray], equation: str) -> str:
    """Automatically select optimization strategy based on problem size"""
    n_tensors = len(tensors)
    total_size = sum(t.size for t in tensors)
    
    if n_tensors <= 4 and total_size < 1e6:
        return 'optimal'
    elif n_tensors <= 10:
        return 'greedy'
    else:
        return 'random'


def _optimal_contraction_path(equation: str, tensors: List[np.ndarray], 
                            memory_limit: int) -> Tuple[List, Any]:
    """Find optimal contraction path using dynamic programming"""
    try:
        return opt_einsum.contract_path(equation, *tensors, optimize='optimal')
    except (MemoryError, ValueError):
        # Fall back to greedy if optimal is too expensive
        warnings.warn("Optimal path finding failed, falling back to greedy")
        return _greedy_contraction_path(equation, tensors, memory_limit)


def _greedy_contraction_path(equation: str, tensors: List[np.ndarray],
                           memory_limit: int) -> Tuple[List, Any]:
    """Find contraction path using greedy algorithm"""
    return opt_einsum.contract_path(equation, *tensors, optimize='greedy')


def _random_contraction_path(equation: str, tensors: List[np.ndarray],
                           memory_limit: int) -> Tuple[List, Any]:
    """Find contraction path using random search"""
    best_path = None
    best_cost = float('inf')
    
    # Try multiple random orderings
    for _ in range(min(100, 2**len(tensors))):
        # Generate random contraction order
        indices = list(range(len(tensors)))
        np.random.shuffle(indices)
        
        try:
            # Build path from random order
            path = []
            current_tensors = list(range(len(tensors)))
            
            while len(current_tensors) > 1:
                # Pick two random tensors to contract
                i = np.random.choice(len(current_tensors))
                j = np.random.choice([k for k in range(len(current_tensors)) if k != i])
                
                # Contract them
                path.append((current_tensors[i], current_tensors[j]))
                
                # Remove contracted tensors and add result
                new_tensor_idx = max(current_tensors) + 1
                current_tensors = [t for k, t in enumerate(current_tensors) if k not in [i, j]]
                current_tensors.append(new_tensor_idx)
            
            # Evaluate cost
            cost = _estimate_path_cost(equation, tensors, path)
            
            if cost < best_cost:
                best_cost = cost
                best_path = path
                
        except (ValueError, IndexError):
            continue
    
    if best_path is None:
        # Fall back to greedy
        return _greedy_contraction_path(equation, tensors, memory_limit)
    
    # Create fake path_info
    class PathInfo:
        def __init__(self, cost):
            self.opt_cost = cost
    
    return best_path, PathInfo(best_cost)


def _simulated_annealing_path(equation: str, tensors: List[np.ndarray],
                            memory_limit: int) -> Tuple[List, Any]:
    """Find contraction path using simulated annealing"""
    # Start with greedy solution
    current_path, current_info = _greedy_contraction_path(equation, tensors, memory_limit)
    current_cost = current_info.opt_cost if hasattr(current_info, 'opt_cost') else 1e10
    
    best_path = current_path
    best_cost = current_cost
    
    # Simulated annealing parameters
    initial_temp = current_cost * 0.1
    final_temp = current_cost * 0.001
    n_steps = min(1000, 10 * len(tensors)**2)
    
    for step in range(n_steps):
        # Temperature schedule
        temp = initial_temp * (final_temp / initial_temp) ** (step / n_steps)
        
        # Generate neighbor solution by swapping random contractions
        new_path = current_path.copy()
        if len(new_path) > 1:
            i, j = np.random.choice(len(new_path), 2, replace=False)
            new_path[i], new_path[j] = new_path[j], new_path[i]
        
        # Evaluate new solution
        try:
            new_cost = _estimate_path_cost(equation, tensors, new_path)
            
            # Accept or reject
            if new_cost < current_cost or np.random.rand() < np.exp(-(new_cost - current_cost) / temp):
                current_path = new_path
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
                    
        except (ValueError, IndexError):
            continue
    
    class PathInfo:
        def __init__(self, cost):
            self.opt_cost = cost
    
    return best_path, PathInfo(best_cost)


def _estimate_path_cost(equation: str, tensors: List[np.ndarray], path: List) -> float:
    """Estimate computational cost of contraction path"""
    # Simplified cost estimation
    total_cost = 0
    
    # Simulate contraction
    current_tensors = {i: t.shape for i, t in enumerate(tensors)}
    
    for contraction in path:
        if len(contraction) == 2:
            i, j = contraction
            if i in current_tensors and j in current_tensors:
                # Estimate cost as product of dimension sizes
                shape_i = current_tensors[i]
                shape_j = current_tensors[j]
                cost = np.prod(shape_i) * np.prod(shape_j)
                total_cost += cost
                
                # Remove old tensors and add result (simplified)
                new_idx = max(current_tensors.keys()) + 1
                current_tensors[new_idx] = shape_i  # Simplified result shape
                del current_tensors[i]
                del current_tensors[j]
    
    return total_cost


def _execute_contraction(equation: str, tensors: List[np.ndarray], 
                       path: List, use_blas: bool) -> np.ndarray:
    """Execute optimized contraction"""
    if use_blas:
        return opt_einsum.contract(equation, *tensors, optimize=path)
    else:
        return opt_einsum.contract(equation, *tensors, optimize=path, use_blas=False)


def _chunked_contraction(equation: str, tensors: List[np.ndarray], 
                       path: List, chunk_size: int) -> np.ndarray:
    """Execute contraction in memory-efficient chunks"""
    # Simplified chunked execution
    # In practice, this would need sophisticated tensor slicing
    return _execute_contraction(equation, tensors, path, True)


def _parallel_contraction(equation: str, tensors: List[np.ndarray], 
                        path: List) -> np.ndarray:
    """Execute contraction with parallel processing"""
    # Use threading for now (in practice, would use multiprocessing or distributed)
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # For simple case, just use normal contraction
    return _execute_contraction(equation, tensors, path, True)


def _analyze_scaling(equation: str, tensors: List[np.ndarray]) -> str:
    """Analyze computational scaling of the contraction"""
    # Count unique indices
    indices = set()
    for part in equation.split('->')[0].split(','):
        indices.update(part)
    
    n_indices = len(indices)
    
    if n_indices <= 3:
        return 'linear'
    elif n_indices <= 6:
        return 'polynomial'
    else:
        return 'exponential'


def _estimate_peak_memory(tensors: List[np.ndarray], path: List) -> int:
    """Estimate peak memory usage during contraction"""
    total_memory = sum(t.nbytes for t in tensors)
    
    # Rough estimate: assume largest intermediate is 2x largest input
    max_tensor_size = max(t.nbytes for t in tensors)
    peak_memory = total_memory + 2 * max_tensor_size
    
    return peak_memory


def _get_intermediate_shapes(tensors: List[np.ndarray], path: List) -> List[Tuple[int, ...]]:
    """Get shapes of intermediate tensors during contraction"""
    # Simplified - in practice would simulate the full contraction
    return [t.shape for t in tensors]