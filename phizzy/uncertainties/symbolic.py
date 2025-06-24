import sympy as sp
import numpy as np
from typing import Union, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class UncertainValue:
    """Represents a value with uncertainty"""
    value: float
    uncertainty: float
    symbol: Optional[sp.Symbol] = None
    
    def __repr__(self):
        return f"{self.value} ± {self.uncertainty}"


@dataclass 
class PropagationResult:
    """Result of uncertainty propagation"""
    expression: sp.Expr
    symbolic_uncertainty: sp.Expr
    numeric_result: UncertainValue
    correlations: Dict[Tuple[str, str], float]
    sensitivity_coefficients: Dict[str, float]
    relative_contributions: Dict[str, float]
    info: Dict[str, Any]


def propagate_uncertainties_symbolic(
    expression: Union[str, sp.Expr],
    variables: Dict[str, Union[UncertainValue, Tuple[float, float]]],
    correlations: Optional[Dict[Tuple[str, str], float]] = None,
    method: str = 'taylor',
    monte_carlo_samples: int = 10000,
    confidence_level: float = 0.68,
    simplify: bool = True
) -> PropagationResult:
    """
    Propagate uncertainties through symbolic expressions with automatic error propagation.
    
    Parameters
    ----------
    expression : str or sympy expression
        Mathematical expression as string or SymPy expression
    variables : dict
        Dictionary mapping variable names to UncertainValue objects or (value, uncertainty) tuples
    correlations : dict, optional
        Correlation coefficients between variables {('x', 'y'): 0.5}
    method : str
        Propagation method: 'taylor' (default), 'monte_carlo', 'bootstrap'
    monte_carlo_samples : int
        Number of samples for Monte Carlo method
    confidence_level : float
        Confidence level for uncertainty (default 0.68 for 1-sigma)
    simplify : bool
        Whether to simplify symbolic expressions
        
    Returns
    -------
    PropagationResult
        Object containing symbolic and numeric results with full diagnostics
    """
    
    # Parse expression if string
    if isinstance(expression, str):
        # Create symbols for all variables
        symbols = {name: sp.Symbol(name) for name in variables}
        local_dict = symbols.copy()
        local_dict.update({
            'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos,
            'tan': sp.tan, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E
        })
        expression = sp.parse_expr(expression, local_dict=local_dict)
    
    # Convert variables to UncertainValue objects
    uncertain_vars = {}
    for name, var in variables.items():
        if isinstance(var, tuple):
            uncertain_vars[name] = UncertainValue(var[0], var[1], sp.Symbol(name))
        else:
            var.symbol = sp.Symbol(name)
            uncertain_vars[name] = var
    
    # Set default correlations
    if correlations is None:
        correlations = {}
    
    # Validate correlations
    for (var1, var2), corr in correlations.items():
        if var1 not in variables or var2 not in variables:
            raise ValueError(f"Correlation between unknown variables: {var1}, {var2}")
        if not -1 <= corr <= 1:
            raise ValueError(f"Correlation must be between -1 and 1, got {corr}")
    
    # Choose propagation method
    if method == 'taylor':
        result = _taylor_propagation(expression, uncertain_vars, correlations, simplify)
    elif method == 'monte_carlo':
        result = _monte_carlo_propagation(
            expression, uncertain_vars, correlations, 
            monte_carlo_samples, confidence_level
        )
    elif method == 'bootstrap':
        result = _bootstrap_propagation(
            expression, uncertain_vars, correlations,
            monte_carlo_samples, confidence_level
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def _taylor_propagation(
    expression: sp.Expr,
    variables: Dict[str, UncertainValue],
    correlations: Dict[Tuple[str, str], float],
    simplify: bool
) -> PropagationResult:
    """First-order Taylor series propagation"""
    
    # Compute partial derivatives
    derivatives = {}
    for name, var in variables.items():
        derivatives[name] = sp.diff(expression, var.symbol)
    
    # Build covariance matrix
    var_names = list(variables.keys())
    n = len(var_names)
    cov_matrix = np.zeros((n, n))
    
    for i, name1 in enumerate(var_names):
        for j, name2 in enumerate(var_names):
            if i == j:
                cov_matrix[i, j] = variables[name1].uncertainty ** 2
            else:
                key = (name1, name2) if (name1, name2) in correlations else (name2, name1)
                if key in correlations:
                    cov_matrix[i, j] = (correlations[key] * 
                                       variables[name1].uncertainty * 
                                       variables[name2].uncertainty)
    
    # Symbolic uncertainty expression
    uncertainty_expr = sp.sqrt(sum(
        sum(derivatives[var_names[i]] * derivatives[var_names[j]] * 
            sp.Symbol(f'cov_{var_names[i]}_{var_names[j]}')
            for j in range(n))
        for i in range(n)
    ))
    
    if simplify:
        uncertainty_expr = sp.simplify(uncertainty_expr)
    
    # Numerical evaluation
    subs_dict = {var.symbol: var.value for var in variables.values()}
    central_value = float(expression.subs(subs_dict))
    
    # Compute sensitivity coefficients
    sensitivity = {}
    relative_contributions = {}
    total_variance = 0
    
    for i, name in enumerate(var_names):
        deriv_value = float(derivatives[name].subs(subs_dict))
        sensitivity[name] = deriv_value
        
        # Contribution to variance
        contrib = deriv_value ** 2 * cov_matrix[i, i]
        for j in range(n):
            if i != j:
                contrib += deriv_value * float(derivatives[var_names[j]].subs(subs_dict)) * cov_matrix[i, j]
        
        total_variance += contrib
        relative_contributions[name] = contrib
    
    # Normalize relative contributions
    if total_variance > 0:
        for name in relative_contributions:
            relative_contributions[name] /= total_variance
    
    uncertainty = np.sqrt(total_variance)
    
    return PropagationResult(
        expression=expression,
        symbolic_uncertainty=uncertainty_expr,
        numeric_result=UncertainValue(central_value, uncertainty),
        correlations=correlations,
        sensitivity_coefficients=sensitivity,
        relative_contributions=relative_contributions,
        info={
            'method': 'taylor',
            'covariance_matrix': cov_matrix,
            'derivatives': derivatives,
            'simplified': simplify
        }
    )


def _monte_carlo_propagation(
    expression: sp.Expr,
    variables: Dict[str, UncertainValue],
    correlations: Dict[Tuple[str, str], float],
    n_samples: int,
    confidence_level: float
) -> PropagationResult:
    """Monte Carlo uncertainty propagation"""
    
    var_names = list(variables.keys())
    n_vars = len(var_names)
    
    # Build correlation matrix
    corr_matrix = np.eye(n_vars)
    for i, name1 in enumerate(var_names):
        for j, name2 in enumerate(var_names):
            if i != j:
                key = (name1, name2) if (name1, name2) in correlations else (name2, name1)
                if key in correlations:
                    corr_matrix[i, j] = correlations[key]
    
    # Generate correlated samples
    means = [variables[name].value for name in var_names]
    stds = [variables[name].uncertainty for name in var_names]
    
    # Convert correlation to covariance
    cov_matrix = corr_matrix * np.outer(stds, stds)
    
    # Generate samples
    samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
    
    # Evaluate expression for each sample
    func = sp.lambdify([var.symbol for var in variables.values()], expression, 'numpy')
    results = np.array([func(*sample) for sample in samples])
    
    # Calculate statistics
    central_value = np.mean(results)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(results, 100 * alpha / 2)
    upper = np.percentile(results, 100 * (1 - alpha / 2))
    uncertainty = (upper - lower) / 2
    
    # Sensitivity analysis via correlation
    sensitivity = {}
    relative_contributions = {}
    
    for i, name in enumerate(var_names):
        corr_coef = np.corrcoef(samples[:, i], results)[0, 1]
        sensitivity[name] = corr_coef
        
        # Approximate relative contribution
        contrib = corr_coef ** 2
        relative_contributions[name] = contrib
    
    # Normalize contributions
    total = sum(relative_contributions.values())
    if total > 0:
        for name in relative_contributions:
            relative_contributions[name] /= total
    
    # Create symbolic uncertainty (approximate)
    uncertainty_expr = sp.sqrt(sum(
        (variables[name].uncertainty * sp.Symbol(f'sens_{name}')) ** 2
        for name in var_names
    ))
    
    return PropagationResult(
        expression=expression,
        symbolic_uncertainty=uncertainty_expr,
        numeric_result=UncertainValue(central_value, uncertainty),
        correlations=correlations,
        sensitivity_coefficients=sensitivity,
        relative_contributions=relative_contributions,
        info={
            'method': 'monte_carlo',
            'n_samples': n_samples,
            'confidence_level': confidence_level,
            'percentiles': {'lower': lower, 'upper': upper},
            'std_dev': np.std(results),
            'skewness': float(sp.stats.skew(results)),
            'kurtosis': float(sp.stats.kurtosis(results))
        }
    )


def _bootstrap_propagation(
    expression: sp.Expr,
    variables: Dict[str, UncertainValue],
    correlations: Dict[Tuple[str, str], float],
    n_samples: int,
    confidence_level: float
) -> PropagationResult:
    """Bootstrap uncertainty propagation with resampling"""
    
    # First get base Monte Carlo result
    mc_result = _monte_carlo_propagation(
        expression, variables, correlations, n_samples, confidence_level
    )
    
    # Perform bootstrap resampling
    var_names = list(variables.keys())
    n_vars = len(var_names)
    n_bootstrap = 1000
    
    # Build correlation matrix
    corr_matrix = np.eye(n_vars)
    for i, name1 in enumerate(var_names):
        for j, name2 in enumerate(var_names):
            if i != j:
                key = (name1, name2) if (name1, name2) in correlations else (name2, name1)
                if key in correlations:
                    corr_matrix[i, j] = correlations[key]
    
    means = [variables[name].value for name in var_names]
    stds = [variables[name].uncertainty for name in var_names]
    cov_matrix = corr_matrix * np.outer(stds, stds)
    
    # Generate bootstrap samples
    func = sp.lambdify([var.symbol for var in variables.values()], expression, 'numpy')
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
        results = np.array([func(*sample) for sample in samples])
        bootstrap_means.append(np.mean(results))
    
    # Bootstrap confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    uncertainty = (upper - lower) / 2
    
    # Update result with bootstrap info
    mc_result.numeric_result.uncertainty = uncertainty
    mc_result.info['method'] = 'bootstrap'
    mc_result.info['n_bootstrap'] = n_bootstrap
    mc_result.info['bootstrap_ci'] = {'lower': lower, 'upper': upper}
    mc_result.info['bootstrap_std'] = np.std(bootstrap_means)
    
    return mc_result