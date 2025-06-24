import numpy as np
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit, differential_evolution
from scipy import stats
import warnings


@dataclass
class PhysicsFitResult:
    """Result of physics-aware experimental fitting"""
    parameters: Dict[str, float]
    parameter_errors: Dict[str, float]
    covariance_matrix: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    p_value: float
    degrees_of_freedom: int
    aic: float
    bic: float
    residuals: np.ndarray
    fitted_curve: np.ndarray
    goodness_of_fit: Dict[str, float]
    systematic_uncertainties: Dict[str, float]
    
    def confidence_interval(self, parameter: str, confidence: float = 0.68) -> Tuple[float, float]:
        """Get confidence interval for parameter"""
        if parameter not in self.parameters:
            raise ValueError(f"Parameter {parameter} not found")
        
        value = self.parameters[parameter]
        error = self.parameter_errors[parameter]
        
        # Use normal approximation
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * error
        
        return (value - margin, value + margin)


def experimental_fit_physics(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model: Union[str, Callable],
    y_errors: Optional[np.ndarray] = None,
    x_errors: Optional[np.ndarray] = None,
    initial_guess: Optional[Dict[str, float]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    systematic_errors: Optional[Dict[str, float]] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    robust_fitting: bool = False,
    distribution: str = 'gaussian',
    background_model: Optional[Union[str, Callable]] = None,
    convolution_kernel: Optional[np.ndarray] = None,
    outlier_detection: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.68
) -> PhysicsFitResult:
    """
    Physics-aware fitting that understands common experimental distributions.
    
    Parameters
    ----------
    x_data : array
        Independent variable data
    y_data : array
        Dependent variable data
    model : str or callable
        Physical model function name or callable
    y_errors : array, optional
        Uncertainties in y_data
    x_errors : array, optional
        Uncertainties in x_data (not yet implemented)
    initial_guess : dict, optional
        Initial parameter guesses {param_name: value}
    parameter_bounds : dict, optional
        Parameter bounds {param_name: (min, max)}
    systematic_errors : dict, optional
        Systematic uncertainties {source: fractional_error}
    correlation_matrix : array, optional
        Correlation matrix for data points
    robust_fitting : bool
        Use robust fitting methods
    distribution : str
        Error distribution: 'gaussian', 'poisson', 'binomial'
    background_model : str or callable, optional
        Background subtraction model
    convolution_kernel : array, optional
        Instrumental response function
    outlier_detection : bool
        Automatically detect and handle outliers
    bootstrap_samples : int
        Number of bootstrap samples for error estimation
    confidence_level : float
        Confidence level for parameter uncertainties
        
    Returns
    -------
    PhysicsFitResult
        Comprehensive fitting results with physics-aware diagnostics
    """
    
    # Parse model
    if isinstance(model, str):
        model_func, param_names = _get_predefined_model(model)
    else:
        model_func = model
        param_names = _extract_parameter_names(model)
    
    # Handle errors
    if y_errors is None:
        if distribution == 'poisson':
            y_errors = np.sqrt(np.maximum(y_data, 1))  # Poisson statistics
        else:
            y_errors = np.ones_like(y_data)  # Uniform weighting
    
    # Outlier detection
    if outlier_detection:
        outlier_mask = _detect_outliers(x_data, y_data, y_errors)
        x_fit = x_data[~outlier_mask]
        y_fit = y_data[~outlier_mask]
        yerr_fit = y_errors[~outlier_mask]
    else:
        x_fit = x_data
        y_fit = y_data
        yerr_fit = y_errors
        outlier_mask = np.zeros(len(x_data), dtype=bool)
    
    # Background subtraction
    if background_model is not None:
        y_fit = _subtract_background(x_fit, y_fit, background_model)
    
    # Set up initial guess
    if initial_guess is None:
        initial_guess = _estimate_initial_parameters(model, x_fit, y_fit, param_names)
    
    # Convert to arrays for fitting
    p0 = [initial_guess.get(name, 1.0) for name in param_names]
    
    # Set up bounds
    if parameter_bounds:
        bounds_lower = [parameter_bounds.get(name, (-np.inf,))[0] for name in param_names]
        bounds_upper = [parameter_bounds.get(name, (np.inf,))[1] for name in param_names]
        bounds = (bounds_lower, bounds_upper)
    else:
        bounds = (-np.inf, np.inf)
    
    # Fitting
    try:
        if robust_fitting:
            # Use robust fitting with outlier-resistant loss function
            result = _robust_fit(model_func, x_fit, y_fit, yerr_fit, p0, bounds, param_names)
        else:
            # Standard least squares
            if distribution == 'gaussian':
                popt, pcov = curve_fit(
                    model_func, x_fit, y_fit, 
                    sigma=yerr_fit, p0=p0, bounds=bounds,
                    absolute_sigma=True
                )
            elif distribution == 'poisson':
                # Maximum likelihood for Poisson
                result = _poisson_fit(model_func, x_fit, y_fit, p0, bounds, param_names)
                popt = result['parameters']
                pcov = result['covariance']
            else:
                # Default to Gaussian
                popt, pcov = curve_fit(model_func, x_fit, y_fit, sigma=yerr_fit, p0=p0)
        
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Fitting failed: {e}")
        # Return reasonable defaults
        return _create_failed_result(param_names, x_data, y_data, str(e))
    
    # Parameter results
    parameters = {name: val for name, val in zip(param_names, popt)}
    parameter_errors = {name: np.sqrt(pcov[i, i]) for i, name in enumerate(param_names)}
    
    # Model evaluation
    fitted_curve = model_func(x_data, *popt)
    
    # Apply convolution if specified
    if convolution_kernel is not None:
        fitted_curve = np.convolve(fitted_curve, convolution_kernel, mode='same')
    
    # Residuals and statistics
    residuals = y_data - fitted_curve
    weights = 1.0 / y_errors**2
    chi_squared = np.sum(weights * residuals**2)
    degrees_of_freedom = len(y_data) - len(popt)
    reduced_chi_squared = chi_squared / degrees_of_freedom
    
    # P-value
    p_value = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
    
    # Information criteria
    n_data = len(y_data)
    n_params = len(popt)
    log_likelihood = -0.5 * chi_squared  # Assuming Gaussian errors
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_data) * n_params - 2 * log_likelihood
    
    # Goodness of fit metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    goodness_of_fit = {
        'r_squared': r_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'p_value': p_value,
        'rmse': np.sqrt(np.mean(residuals**2))
    }
    
    # Systematic uncertainties
    systematic_uncertainties = {}
    if systematic_errors:
        for source, fractional_error in systematic_errors.items():
            for name in param_names:
                sys_error = abs(parameters[name] * fractional_error)
                systematic_uncertainties[f'{name}_{source}'] = sys_error
    
    # Bootstrap error estimation if requested
    if bootstrap_samples > 0:
        bootstrap_errors = _bootstrap_errors(
            model_func, x_fit, y_fit, yerr_fit, popt, param_names, bootstrap_samples
        )
        # Update parameter errors with bootstrap results
        for name in param_names:
            if name in bootstrap_errors:
                parameter_errors[name] = max(parameter_errors[name], bootstrap_errors[name])
    
    return PhysicsFitResult(
        parameters=parameters,
        parameter_errors=parameter_errors,
        covariance_matrix=pcov,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        p_value=p_value,
        degrees_of_freedom=degrees_of_freedom,
        aic=aic,
        bic=bic,
        residuals=residuals,
        fitted_curve=fitted_curve,
        goodness_of_fit=goodness_of_fit,
        systematic_uncertainties=systematic_uncertainties
    )


def _get_predefined_model(model_name: str) -> Tuple[Callable, List[str]]:
    """Get predefined physics model functions"""
    
    models = {
        'linear': (lambda x, a, b: a * x + b, ['a', 'b']),
        'exponential': (lambda x, a, b, c: a * np.exp(-b * x) + c, ['a', 'b', 'c']),
        'gaussian': (lambda x, a, mu, sigma: a * np.exp(-(x - mu)**2 / (2 * sigma**2)), 
                    ['a', 'mu', 'sigma']),
        'lorentzian': (lambda x, a, x0, gamma: a * gamma**2 / ((x - x0)**2 + gamma**2),
                      ['a', 'x0', 'gamma']),
        'power_law': (lambda x, a, b: a * x**b, ['a', 'b']),
        'sine': (lambda x, a, f, phi, c: a * np.sin(2 * np.pi * f * x + phi) + c,
                ['a', 'f', 'phi', 'c']),
        'cosine': (lambda x, a, f, phi, c: a * np.cos(2 * np.pi * f * x + phi) + c,
                  ['a', 'f', 'phi', 'c']),
        'damped_oscillator': (lambda x, a, f, gamma, phi: a * np.exp(-gamma * x) * 
                             np.cos(2 * np.pi * f * x + phi),
                             ['a', 'f', 'gamma', 'phi']),
        'fermi_dirac': (lambda x, a, mu, kT: a / (1 + np.exp((x - mu) / kT)),
                       ['a', 'mu', 'kT']),
        'bose_einstein': (lambda x, a, mu, kT: a / (np.exp((x - mu) / kT) - 1),
                         ['a', 'mu', 'kT']),
        'arrhenius': (lambda x, a, ea, R: a * np.exp(-ea / (R * x)), ['a', 'ea', 'R'])
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]


def _extract_parameter_names(func: Callable) -> List[str]:
    """Extract parameter names from function signature"""
    import inspect
    sig = inspect.signature(func)
    return [name for name in sig.parameters.keys() if name != 'x']


def _detect_outliers(x_data: np.ndarray, y_data: np.ndarray, y_errors: np.ndarray) -> np.ndarray:
    """Detect outliers using modified Z-score"""
    # Simple outlier detection based on residuals from median
    median_y = np.median(y_data)
    mad = np.median(np.abs(y_data - median_y))
    
    # Modified Z-score
    modified_z_scores = 0.6745 * (y_data - median_y) / mad
    
    # Outliers are points with |modified Z-score| > 3.5
    return np.abs(modified_z_scores) > 3.5


def _subtract_background(x_data: np.ndarray, y_data: np.ndarray, 
                        background_model: Union[str, Callable]) -> np.ndarray:
    """Subtract background from data"""
    if isinstance(background_model, str):
        if background_model == 'linear':
            # Fit linear background to edges of data
            edge_fraction = 0.1
            n_edge = int(len(x_data) * edge_fraction)
            
            x_bg = np.concatenate([x_data[:n_edge], x_data[-n_edge:]])
            y_bg = np.concatenate([y_data[:n_edge], y_data[-n_edge:]])
            
            # Linear fit
            p = np.polyfit(x_bg, y_bg, 1)
            background = np.polyval(p, x_data)
        elif background_model == 'constant':
            # Use minimum as background
            background = np.full_like(y_data, np.min(y_data))
        else:
            background = np.zeros_like(y_data)
    else:
        # Custom background function
        # Would need to fit this first
        background = np.zeros_like(y_data)
    
    return y_data - background


def _estimate_initial_parameters(model: str, x_data: np.ndarray, y_data: np.ndarray,
                                param_names: List[str]) -> Dict[str, float]:
    """Estimate reasonable initial parameters"""
    
    initial_guess = {}
    
    # Generic estimates based on data characteristics
    y_range = np.max(y_data) - np.min(y_data)
    x_range = np.max(x_data) - np.min(x_data)
    y_mean = np.mean(y_data)
    
    for name in param_names:
        if name in ['a', 'amplitude']:
            initial_guess[name] = y_range
        elif name in ['b', 'slope']:
            initial_guess[name] = y_range / x_range
        elif name in ['c', 'offset', 'background']:
            initial_guess[name] = np.min(y_data)
        elif name in ['mu', 'x0', 'center']:
            initial_guess[name] = x_data[np.argmax(y_data)]
        elif name in ['sigma', 'width', 'gamma']:
            initial_guess[name] = x_range / 10
        elif name in ['f', 'frequency']:
            initial_guess[name] = 1.0 / x_range
        else:
            initial_guess[name] = 1.0
    
    return initial_guess


def _robust_fit(model_func: Callable, x_data: np.ndarray, y_data: np.ndarray,
               y_errors: np.ndarray, p0: List[float], bounds: Tuple, 
               param_names: List[str]) -> Dict:
    """Robust fitting using loss functions less sensitive to outliers"""
    
    def robust_loss(params):
        y_model = model_func(x_data, *params)
        residuals = (y_data - y_model) / y_errors
        
        # Huber loss (less sensitive to outliers than squared loss)
        delta = 1.345  # Standard value
        loss = np.where(np.abs(residuals) <= delta,
                       0.5 * residuals**2,
                       delta * (np.abs(residuals) - 0.5 * delta))
        return np.sum(loss)
    
    # Use differential evolution for robust global optimization
    result = differential_evolution(robust_loss, bounds=list(zip(*bounds)), seed=42)
    
    # Estimate covariance matrix
    popt = result.x
    y_model = model_func(x_data, *popt)
    residuals = y_data - y_model
    
    # Approximate covariance using Fisher information
    n_params = len(popt)
    pcov = np.eye(n_params) * np.var(residuals)  # Simplified estimate
    
    return {'parameters': popt, 'covariance': pcov}


def _poisson_fit(model_func: Callable, x_data: np.ndarray, y_data: np.ndarray,
                p0: List[float], bounds: Tuple, param_names: List[str]) -> Dict:
    """Maximum likelihood fitting for Poisson statistics"""
    
    def negative_log_likelihood(params):
        y_model = model_func(x_data, *params)
        # Ensure positive model values
        y_model = np.maximum(y_model, 1e-10)
        
        # Poisson log-likelihood
        log_likelihood = np.sum(y_data * np.log(y_model) - y_model)
        return -log_likelihood
    
    result = differential_evolution(negative_log_likelihood, bounds=list(zip(*bounds)), seed=42)
    
    popt = result.x
    
    # Estimate covariance matrix using Fisher information
    epsilon = 1e-8
    n_params = len(popt)
    fisher_matrix = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            # Numerical second derivative
            params_ij = popt.copy()
            params_ij[i] += epsilon
            params_ij[j] += epsilon
            
            f_ij = negative_log_likelihood(params_ij)
            f_i = negative_log_likelihood(popt + epsilon * np.eye(n_params)[i])
            f_j = negative_log_likelihood(popt + epsilon * np.eye(n_params)[j])
            f_0 = negative_log_likelihood(popt)
            
            fisher_matrix[i, j] = (f_ij - f_i - f_j + f_0) / (epsilon**2)
    
    # Covariance is inverse of Fisher information matrix
    try:
        pcov = np.linalg.inv(fisher_matrix)
    except np.linalg.LinAlgError:
        pcov = np.eye(n_params) * 0.1  # Fallback
    
    return {'parameters': popt, 'covariance': pcov}


def _bootstrap_errors(model_func: Callable, x_data: np.ndarray, y_data: np.ndarray,
                     y_errors: np.ndarray, popt: np.ndarray, param_names: List[str],
                     n_bootstrap: int) -> Dict[str, float]:
    """Estimate parameter uncertainties using bootstrap resampling"""
    
    bootstrap_params = []
    
    for _ in range(min(n_bootstrap, 100)):  # Limit for performance
        # Resample with replacement
        indices = np.random.choice(len(x_data), len(x_data), replace=True)
        x_boot = x_data[indices]
        y_boot = y_data[indices]
        yerr_boot = y_errors[indices]
        
        try:
            popt_boot, _ = curve_fit(model_func, x_boot, y_boot, sigma=yerr_boot, p0=popt)
            bootstrap_params.append(popt_boot)
        except:
            continue
    
    if not bootstrap_params:
        return {}
    
    bootstrap_params = np.array(bootstrap_params)
    bootstrap_errors = {}
    
    for i, name in enumerate(param_names):
        bootstrap_errors[name] = np.std(bootstrap_params[:, i])
    
    return bootstrap_errors


def _create_failed_result(param_names: List[str], x_data: np.ndarray, 
                         y_data: np.ndarray, error_message: str) -> PhysicsFitResult:
    """Create a result object for failed fits"""
    
    n_params = len(param_names)
    
    return PhysicsFitResult(
        parameters={name: np.nan for name in param_names},
        parameter_errors={name: np.inf for name in param_names},
        covariance_matrix=np.full((n_params, n_params), np.nan),
        chi_squared=np.inf,
        reduced_chi_squared=np.inf,
        p_value=0.0,
        degrees_of_freedom=len(y_data) - n_params,
        aic=np.inf,
        bic=np.inf,
        residuals=np.full_like(y_data, np.nan),
        fitted_curve=np.full_like(y_data, np.nan),
        goodness_of_fit={'error': error_message},
        systematic_uncertainties={}
    )