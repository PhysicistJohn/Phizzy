import sympy as sp
import numpy as np
from typing import List, Dict, Union, Callable, Optional, Any
from dataclasses import dataclass


@dataclass
class LagrangianResult:
    """Result of Lagrangian analysis"""
    equations_of_motion: List[sp.Expr]
    generalized_coords: List[sp.Symbol]
    generalized_velocities: List[sp.Symbol]
    lagrangian: sp.Expr
    kinetic_energy: sp.Expr
    potential_energy: sp.Expr
    constraints: List[sp.Expr]
    lagrange_multipliers: List[sp.Symbol]
    conserved_quantities: List[sp.Expr]
    
    def get_numerical_function(self) -> Callable:
        """Convert to numerical function for integration"""
        return sp.lambdify(
            self.generalized_coords + self.generalized_velocities,
            self.equations_of_motion,
            'numpy'
        )


def lagrangian_to_equations(
    lagrangian: Union[str, sp.Expr, Callable],
    coordinates: List[Union[str, sp.Symbol]],
    time_symbol: Union[str, sp.Symbol] = 't',
    constraints: Optional[List[Union[str, sp.Expr]]] = None,
    potential_energy: Optional[Union[str, sp.Expr]] = None,
    kinetic_energy: Optional[Union[str, sp.Expr]] = None,
    external_forces: Optional[Dict[str, Union[str, sp.Expr]]] = None,
    dissipation_function: Optional[Union[str, sp.Expr]] = None,
    conserved_quantities: bool = True,
    simplify_equations: bool = True
) -> LagrangianResult:
    """
    Convert Lagrangian/Hamiltonian to equations of motion with automatic constraint handling.
    
    Parameters
    ----------
    lagrangian : str, sympy expr, or callable
        Lagrangian function L(q, q_dot, t) or expression
    coordinates : list
        Generalized coordinates (position variables)
    time_symbol : str or symbol
        Time variable
    constraints : list, optional
        Constraint equations (holonomic constraints)
    potential_energy : str or expr, optional
        Potential energy V(q) if separate from Lagrangian
    kinetic_energy : str or expr, optional
        Kinetic energy T(q, q_dot) if separate from Lagrangian
    external_forces : dict, optional
        Non-conservative forces {coord: force_expr}
    dissipation_function : str or expr, optional
        Rayleigh dissipation function D(q_dot)
    conserved_quantities : bool
        Find conserved quantities (energy, momentum, etc.)
    simplify_equations : bool
        Simplify resulting equations
        
    Returns
    -------
    LagrangianResult
        Equations of motion and analysis results
    """
    
    # Convert inputs to SymPy symbols
    if isinstance(time_symbol, str):
        t = sp.Symbol(time_symbol)
    else:
        t = time_symbol
    
    # Convert coordinates to symbols
    q_symbols = []
    for coord in coordinates:
        if isinstance(coord, str):
            q_symbols.append(sp.Symbol(coord))
        else:
            q_symbols.append(coord)
    
    # Create velocity symbols
    q_dot_symbols = [sp.Symbol(f'{q.name}_dot') for q in q_symbols]
    
    # Create time-dependent functions for coordinates and velocities
    q_funcs = [sp.Function(q.name)(t) for q in q_symbols]
    q_dot_funcs = [sp.diff(q_func, t) for q_func in q_funcs]
    
    # Parse Lagrangian
    if isinstance(lagrangian, str):
        # Create local namespace with coordinates and velocities
        local_dict = {q.name: q_func for q, q_func in zip(q_symbols, q_funcs)}
        local_dict.update({f'{q.name}_dot': q_dot_func for q, q_dot_func in zip(q_symbols, q_dot_funcs)})
        local_dict[t.name] = t
        local_dict.update({
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'pi': sp.pi, 'E': sp.E
        })
        L = sp.parse_expr(lagrangian, local_dict=local_dict)
    elif callable(lagrangian):
        # Convert callable to symbolic
        L = lagrangian(*q_funcs, *q_dot_funcs, t)
    else:
        L = lagrangian
    
    # If kinetic and potential are given separately, combine them
    if kinetic_energy is not None and potential_energy is not None:
        if isinstance(kinetic_energy, str):
            T = sp.parse_expr(kinetic_energy, local_dict=local_dict)
        else:
            T = kinetic_energy
        
        if isinstance(potential_energy, str):
            V = sp.parse_expr(potential_energy, local_dict=local_dict)
        else:
            V = potential_energy
        
        L = T - V
    else:
        # Try to identify T and V from Lagrangian
        T, V = _identify_kinetic_potential(L, q_funcs, q_dot_funcs)
    
    # Apply Euler-Lagrange equations
    equations = []
    for i, (q_func, q_dot_func) in enumerate(zip(q_funcs, q_dot_funcs)):
        # Euler-Lagrange: d/dt(∂L/∂q_dot) - ∂L/∂q = 0
        
        # ∂L/∂q_dot
        dL_dqdot = sp.diff(L, q_dot_func)
        
        # d/dt(∂L/∂q_dot)
        d_dt_dL_dqdot = sp.diff(dL_dqdot, t)
        
        # ∂L/∂q
        dL_dq = sp.diff(L, q_func)
        
        # Equation of motion
        eq = d_dt_dL_dqdot - dL_dq
        
        # Add external forces
        if external_forces and q_symbols[i].name in external_forces:
            force_expr = external_forces[q_symbols[i].name]
            if isinstance(force_expr, str):
                force = sp.parse_expr(force_expr, local_dict=local_dict)
            else:
                force = force_expr
            eq += force
        
        # Add dissipation
        if dissipation_function is not None:
            if isinstance(dissipation_function, str):
                D = sp.parse_expr(dissipation_function, local_dict=local_dict)
            else:
                D = dissipation_function
            
            # Add -∂D/∂q_dot term
            eq -= sp.diff(D, q_dot_func)
        
        equations.append(eq)
    
    # Handle constraints using Lagrange multipliers
    constraint_equations = []
    lambda_symbols = []
    
    if constraints:
        for i, constraint in enumerate(constraints):
            if isinstance(constraint, str):
                constraint_expr = sp.parse_expr(constraint, local_dict=local_dict)
            else:
                constraint_expr = constraint
            
            constraint_equations.append(constraint_expr)
            
            # Add Lagrange multiplier
            lambda_sym = sp.Symbol(f'lambda_{i}')
            lambda_symbols.append(lambda_sym)
            
            # Modify equations of motion
            for j, q_func in enumerate(q_funcs):
                constraint_force = lambda_sym * sp.diff(constraint_expr, q_func)
                equations[j] += constraint_force
    
    # Convert to second-order ODEs
    second_order_eqs = []
    for i, eq in enumerate(equations):
        # Solve for q_ddot (second derivative)
        q_ddot = sp.diff(q_funcs[i], t, 2)
        
        try:
            # Try to solve for q_ddot
            solved = sp.solve(eq, q_ddot)
            if solved:
                second_order_eqs.append(solved[0])
            else:
                # If can't solve, keep original form
                second_order_eqs.append(eq)
        except (sp.SolvingError, NotImplementedError):
            second_order_eqs.append(eq)
    
    # Simplify if requested
    if simplify_equations:
        second_order_eqs = [sp.simplify(eq) for eq in second_order_eqs]
    
    # Find conserved quantities
    conserved_qtys = []
    if conserved_quantities:
        conserved_qtys = _find_conserved_quantities(L, q_funcs, q_dot_funcs, t)
    
    # Convert function form back to symbol form for output
    symbol_mapping = {}
    for q_func, q_sym in zip(q_funcs, q_symbols):
        symbol_mapping[q_func] = q_sym
    for q_dot_func, q_dot_sym in zip(q_dot_funcs, q_dot_symbols):
        symbol_mapping[q_dot_func] = q_dot_sym
    
    # Apply mapping to equations
    final_equations = []
    for eq in second_order_eqs:
        final_eq = eq.subs(symbol_mapping)
        final_equations.append(final_eq)
    
    # Apply mapping to other expressions
    final_L = L.subs(symbol_mapping)
    final_T = T.subs(symbol_mapping) if 'T' in locals() else sp.sympify(0)
    final_V = V.subs(symbol_mapping) if 'V' in locals() else sp.sympify(0)
    
    final_conserved = []
    for qty in conserved_qtys:
        final_conserved.append(qty.subs(symbol_mapping))
    
    return LagrangianResult(
        equations_of_motion=final_equations,
        generalized_coords=q_symbols,
        generalized_velocities=q_dot_symbols,
        lagrangian=final_L,
        kinetic_energy=final_T,
        potential_energy=final_V,
        constraints=constraint_equations,
        lagrange_multipliers=lambda_symbols,
        conserved_quantities=final_conserved
    )


def _identify_kinetic_potential(L: sp.Expr, q_funcs: List, q_dot_funcs: List) -> tuple:
    """Try to identify kinetic and potential energy from Lagrangian"""
    
    # Kinetic energy: terms with q_dot squared
    T_terms = []
    V_terms = []
    
    # This is a simplified heuristic
    # In practice, would need more sophisticated parsing
    
    expanded = sp.expand(L)
    
    if hasattr(expanded, 'args'):
        for term in expanded.args:
            has_velocity = any(q_dot in term.free_symbols for q_dot in q_dot_funcs)
            
            if has_velocity:
                T_terms.append(term)
            else:
                V_terms.append(-term)  # Potential has opposite sign in L = T - V
    else:
        # Single term
        has_velocity = any(q_dot in expanded.free_symbols for q_dot in q_dot_funcs)
        if has_velocity:
            T_terms.append(expanded)
        else:
            V_terms.append(-expanded)
    
    T = sum(T_terms) if T_terms else sp.sympify(0)
    V = sum(V_terms) if V_terms else sp.sympify(0)
    
    return T, V


def _find_conserved_quantities(L: sp.Expr, q_funcs: List, q_dot_funcs: List, t: sp.Symbol) -> List[sp.Expr]:
    """Find conserved quantities using Noether's theorem"""
    
    conserved = []
    
    # Energy conservation (if L doesn't explicitly depend on time)
    if t not in L.free_symbols:
        # Hamiltonian = sum(q_dot * ∂L/∂q_dot) - L
        H = sum(q_dot * sp.diff(L, q_dot) for q_dot in q_dot_funcs) - L
        conserved.append(H)
    
    # Momentum conservation (if L doesn't depend on specific coordinate)
    for i, q_func in enumerate(q_funcs):
        if q_func not in L.free_symbols:
            # Generalized momentum p = ∂L/∂q_dot
            p = sp.diff(L, q_dot_funcs[i])
            conserved.append(p)
    
    # Angular momentum (for rotational symmetry)
    # This is a simplified check for cylindrical/spherical coordinates
    coord_names = [q.func.__name__ for q in q_funcs]
    if 'theta' in coord_names:
        theta_idx = coord_names.index('theta')
        if q_funcs[theta_idx] not in L.free_symbols:
            L_z = sp.diff(L, q_dot_funcs[theta_idx])
            conserved.append(L_z)
    
    return conserved