import numpy as np
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
from scipy.interpolate import UnivariateSpline
import warnings


@dataclass
class PhaseTransitionResult:
    """Result of phase transition detection"""
    transition_points: List[float]
    transition_types: List[str]
    order_parameters: Dict[str, np.ndarray]
    critical_exponents: Dict[str, float]
    confidence_scores: List[float]
    phases: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    
    def get_phase_at(self, parameter_value: float) -> Dict[str, Any]:
        """Get phase information at given parameter value"""
        for i, point in enumerate(self.transition_points):
            if parameter_value < point:
                return self.phases[i] if i < len(self.phases) else self.phases[-1]
        return self.phases[-1] if self.phases else {}


def phase_transition_detect(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    control_parameter: Union[np.ndarray, str] = None,
    observables: Optional[Dict[str, np.ndarray]] = None,
    method: str = 'auto',
    transition_types: List[str] = ['continuous', 'discontinuous'],
    order_parameter: Optional[str] = None,
    finite_size_scaling: bool = True,
    system_sizes: Optional[List[int]] = None,
    confidence_threshold: float = 0.8,
    bootstrap_samples: int = 1000,
    kernel_bandwidth: Optional[float] = None,
    smoothing_factor: float = 0.1
) -> PhaseTransitionResult:
    """
    Automated detection and characterization of phase transitions from simulation or experimental data.
    
    Parameters
    ----------
    data : array or dict
        Time series data or dictionary of observables vs control parameter
    control_parameter : array or str
        Values of control parameter (e.g., temperature) or key in data dict
    observables : dict
        Dictionary of observable arrays
    method : str
        Detection method: 'auto', 'clustering', 'derivatives', 'ml', 'finite-size'
    transition_types : list
        Types to detect: 'continuous', 'discontinuous', 'kosterlitz-thouless'
    order_parameter : str
        Name of primary order parameter
    finite_size_scaling : bool
        Use finite-size scaling analysis
    system_sizes : list
        System sizes for finite-size scaling
    confidence_threshold : float
        Minimum confidence for transition detection
    bootstrap_samples : int
        Number of bootstrap samples for error estimation
    kernel_bandwidth : float
        Bandwidth for kernel-based methods
    smoothing_factor : float
        Smoothing factor for numerical derivatives
        
    Returns
    -------
    PhaseTransitionResult
        Detection results with transition points and characterization
    """
    
    # Parse input data
    if isinstance(data, dict):
        if control_parameter is None:
            control_parameter = list(data.keys())[0]
        if isinstance(control_parameter, str):
            if control_parameter not in data:
                raise ValueError(f"Control parameter '{control_parameter}' not found in data")
            x_values = data[control_parameter]
            observables = {k: v for k, v in data.items() if k != control_parameter}
        else:
            x_values = control_parameter
    else:
        if control_parameter is None:
            x_values = np.arange(len(data))
        else:
            x_values = control_parameter
        
        if observables is None:
            observables = {'main': data}
    
    # Ensure arrays are properly shaped
    x_values = np.atleast_1d(x_values)
    for key in observables:
        observables[key] = np.atleast_1d(observables[key])
    
    # Select detection method
    if method == 'auto':
        method = _auto_select_method(observables, x_values)
    
    # Detect transitions
    if method == 'clustering':
        result = _clustering_detection(x_values, observables, transition_types)
    elif method == 'derivatives':
        result = _derivative_detection(x_values, observables, smoothing_factor)
    elif method == 'ml':
        result = _ml_detection(x_values, observables, transition_types)
    elif method == 'finite-size':
        result = _finite_size_detection(x_values, observables, system_sizes)
    else:
        result = _combined_detection(x_values, observables, transition_types, smoothing_factor)
    
    # Post-process results
    if finite_size_scaling and system_sizes:
        result = _apply_finite_size_scaling(result, system_sizes)
    
    # Bootstrap error estimation
    if bootstrap_samples > 0:
        result = _bootstrap_analysis(result, x_values, observables, bootstrap_samples)
    
    # Filter by confidence
    filtered_transitions = []
    filtered_types = []
    filtered_confidence = []
    
    for i, (point, conf) in enumerate(zip(result.transition_points, result.confidence_scores)):
        if conf >= confidence_threshold:
            filtered_transitions.append(point)
            filtered_types.append(result.transition_types[i])
            filtered_confidence.append(conf)
    
    result.transition_points = filtered_transitions
    result.transition_types = filtered_types
    result.confidence_scores = filtered_confidence
    
    return result


def _auto_select_method(observables: Dict[str, np.ndarray], x_values: np.ndarray) -> str:
    """Automatically select detection method based on data characteristics"""
    n_points = len(x_values)
    n_observables = len(observables)
    
    if n_points < 50:
        return 'clustering'
    elif n_observables == 1:
        return 'derivatives'
    else:
        return 'ml'


def _clustering_detection(x_values: np.ndarray, observables: Dict[str, np.ndarray], 
                         transition_types: List[str]) -> PhaseTransitionResult:
    """Detect transitions using clustering analysis"""
    
    # Combine observables into feature matrix
    features = np.column_stack([obs for obs in observables.values()])
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Try different numbers of clusters
    best_score = -np.inf
    best_n_clusters = 2
    
    for n_clusters in range(2, min(10, len(x_values) // 5)):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            # Use silhouette score as quality metric
            from sklearn.metrics import silhouette_score
            score = silhouette_score(features_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                
        except (ValueError, ImportError):
            continue
    
    # Final clustering with best parameters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    # Find transition points
    transition_points = []
    transition_types = []
    confidence_scores = []
    
    for i in range(best_n_clusters - 1):
        # Find boundary between clusters
        mask_current = labels == i
        mask_next = labels == (i + 1)
        
        if np.any(mask_current) and np.any(mask_next):
            x_current = x_values[mask_current]
            x_next = x_values[mask_next]
            
            # Transition point is at boundary
            transition_point = (np.max(x_current) + np.min(x_next)) / 2
            transition_points.append(transition_point)
            
            # Classify transition type based on feature change
            features_current = np.mean(features[mask_current], axis=0)
            features_next = np.mean(features[mask_next], axis=0)
            
            # Simple heuristic: large jump = discontinuous, gradual = continuous
            feature_change = np.linalg.norm(features_next - features_current)
            feature_scale = np.linalg.norm(features_current) + np.linalg.norm(features_next)
            
            if feature_change / feature_scale > 0.5:
                transition_types.append('discontinuous')
            else:
                transition_types.append('continuous')
            
            confidence_scores.append(best_score)
    
    # Identify phases
    phases = []
    for i in range(best_n_clusters):
        mask = labels == i
        phase_data = {
            'label': i,
            'x_range': (np.min(x_values[mask]), np.max(x_values[mask])),
            'mean_observables': {key: np.mean(obs[mask]) for key, obs in observables.items()},
            'std_observables': {key: np.std(obs[mask]) for key, obs in observables.items()}
        }
        phases.append(phase_data)
    
    return PhaseTransitionResult(
        transition_points=transition_points,
        transition_types=transition_types,
        order_parameters=observables,
        critical_exponents={},
        confidence_scores=confidence_scores,
        phases=phases,
        statistics={'clustering_score': best_score, 'n_clusters': best_n_clusters}
    )


def _derivative_detection(x_values: np.ndarray, observables: Dict[str, np.ndarray],
                         smoothing_factor: float) -> PhaseTransitionResult:
    """Detect transitions using derivative analysis"""
    
    transition_points = []
    transition_types = []
    confidence_scores = []
    critical_exponents = {}
    
    # First try simple jump detection for discontinuous transitions
    for obs_name, obs_data in observables.items():
        # Calculate differences between adjacent points
        diffs = np.diff(obs_data)
        diff_std = np.std(diffs)
        
        # Find large jumps (more than 3 standard deviations)
        jump_indices = np.where(np.abs(diffs) > 3 * diff_std)[0]
        
        for idx in jump_indices:
            if idx > 0 and idx < len(x_values) - 1:
                x_transition = (x_values[idx] + x_values[idx + 1]) / 2
                transition_points.append(x_transition)
                transition_types.append('discontinuous')
                confidence_scores.append(min(np.abs(diffs[idx]) / (5 * diff_std), 1.0))
    
    # Analyze each observable for continuous transitions
    for obs_name, obs_data in observables.items():
        # Smooth data
        if len(x_values) > 4:
            try:
                # Use less aggressive smoothing for better detection
                smooth_factor = smoothing_factor * np.std(obs_data)
                spline = UnivariateSpline(x_values, obs_data, s=smooth_factor)
                obs_smooth = spline(x_values)
                
                # Compute derivatives
                dx = np.mean(np.diff(x_values))  # Use mean spacing
                first_deriv = np.gradient(obs_smooth, dx)
                second_deriv = np.gradient(first_deriv, dx)
                
                # Find peaks in absolute second derivative
                # Use more sensitive threshold for detection
                threshold = 0.5 * np.std(second_deriv)
                min_distance = max(3, len(x_values) // 20)
                
                peaks, properties = signal.find_peaks(
                    np.abs(second_deriv), 
                    height=threshold,
                    distance=min_distance,
                    prominence=threshold
                )
                
                for peak_idx in peaks:
                    x_peak = x_values[peak_idx]
                    
                    # Classify transition type
                    if np.abs(first_deriv[peak_idx]) > 3 * np.std(first_deriv):
                        trans_type = 'discontinuous'
                    else:
                        trans_type = 'continuous'
                    
                    # Estimate critical exponent for continuous transitions
                    if trans_type == 'continuous':
                        exponent = _estimate_critical_exponent(
                            x_values, obs_smooth, x_peak
                        )
                        critical_exponents[f'{obs_name}_exponent'] = exponent
                    
                    confidence = properties['peak_heights'][list(peaks).index(peak_idx)] if 'peak_heights' in properties else 0.5
                    
                    transition_points.append(x_peak)
                    transition_types.append(trans_type)
                    confidence_scores.append(min(confidence / np.max(np.abs(second_deriv)), 1.0))
                    
            except (ValueError, TypeError):
                continue
    
    # Remove duplicates and sort
    if transition_points:
        sorted_indices = np.argsort(transition_points)
        transition_points = [transition_points[i] for i in sorted_indices]
        transition_types = [transition_types[i] for i in sorted_indices]
        confidence_scores = [confidence_scores[i] for i in sorted_indices]
    
    return PhaseTransitionResult(
        transition_points=transition_points,
        transition_types=transition_types,
        order_parameters=observables,
        critical_exponents=critical_exponents,
        confidence_scores=confidence_scores,
        phases=[],
        statistics={'method': 'derivatives'}
    )


def _ml_detection(x_values: np.ndarray, observables: Dict[str, np.ndarray],
                 transition_types: List[str]) -> PhaseTransitionResult:
    """Detect transitions using machine learning approaches"""
    
    # Use Gaussian Mixture Model for unsupervised detection
    features = np.column_stack([obs for obs in observables.values()])
    
    # Try different numbers of components
    best_bic = np.inf
    best_n_components = 2
    
    for n_components in range(2, min(8, len(x_values) // 10)):
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(features)
            bic = gmm.bic(features)
            
            if bic < best_bic:
                best_bic = bic
                best_n_components = n_components
                
        except (ValueError, np.linalg.LinAlgError):
            continue
    
    # Final model
    gmm = GaussianMixture(n_components=best_n_components, random_state=42)
    gmm.fit(features)
    labels = gmm.predict(features)
    probabilities = gmm.predict_proba(features)
    
    # Find transition points
    transition_points = []
    transition_types = []
    confidence_scores = []
    
    # Look for changes in dominant component
    for i in range(len(x_values) - 1):
        if labels[i] != labels[i + 1]:
            # Transition detected
            x_transition = (x_values[i] + x_values[i + 1]) / 2
            
            # Confidence based on probability difference
            prob_diff = np.abs(probabilities[i] - probabilities[i + 1]).max()
            confidence = min(prob_diff * 2, 1.0)
            
            # Type classification based on sharpness
            if confidence > 0.8:
                trans_type = 'discontinuous'
            else:
                trans_type = 'continuous'
            
            transition_points.append(x_transition)
            transition_types.append(trans_type)
            confidence_scores.append(confidence)
    
    return PhaseTransitionResult(
        transition_points=transition_points,
        transition_types=transition_types,
        order_parameters=observables,
        critical_exponents={},
        confidence_scores=confidence_scores,
        phases=[],
        statistics={'bic': best_bic, 'n_components': best_n_components}
    )


def _finite_size_detection(x_values: np.ndarray, observables: Dict[str, np.ndarray],
                          system_sizes: Optional[List[int]]) -> PhaseTransitionResult:
    """Detect transitions using finite-size scaling"""
    
    if system_sizes is None:
        # Try to infer from data structure
        system_sizes = [int(np.sqrt(len(x_values)))]
    
    # This is a simplified implementation
    # In practice, would need multiple datasets for different system sizes
    return _derivative_detection(x_values, observables, 0.1)


def _combined_detection(x_values: np.ndarray, observables: Dict[str, np.ndarray],
                       transition_types: List[str], smoothing_factor: float) -> PhaseTransitionResult:
    """Combine multiple detection methods"""
    
    # Get results from different methods
    clustering_result = _clustering_detection(x_values, observables, transition_types)
    derivative_result = _derivative_detection(x_values, observables, smoothing_factor)
    
    # Combine transition points (simple voting)
    all_points = clustering_result.transition_points + derivative_result.transition_points
    all_confidences = clustering_result.confidence_scores + derivative_result.confidence_scores
    
    if not all_points:
        return clustering_result
    
    # Merge nearby points
    tolerance = (np.max(x_values) - np.min(x_values)) * 0.05
    merged_points = []
    merged_types = []
    merged_confidences = []
    
    sorted_indices = np.argsort(all_points)
    
    for i in sorted_indices:
        point = all_points[i]
        confidence = all_confidences[i]
        
        # Check if close to existing point
        merged = False
        for j, existing_point in enumerate(merged_points):
            if abs(point - existing_point) < tolerance:
                # Merge with existing point
                merged_points[j] = (merged_points[j] * merged_confidences[j] + 
                                  point * confidence) / (merged_confidences[j] + confidence)
                merged_confidences[j] = max(merged_confidences[j], confidence)
                merged = True
                break
        
        if not merged:
            merged_points.append(point)
            merged_types.append('continuous')  # Default
            merged_confidences.append(confidence)
    
    return PhaseTransitionResult(
        transition_points=merged_points,
        transition_types=merged_types,
        order_parameters=observables,
        critical_exponents=derivative_result.critical_exponents,
        confidence_scores=merged_confidences,
        phases=clustering_result.phases,
        statistics={'method': 'combined'}
    )


def _estimate_critical_exponent(x_values: np.ndarray, obs_data: np.ndarray, 
                               x_critical: float) -> float:
    """Estimate critical exponent near transition"""
    
    # Find points near critical point
    distance = np.abs(x_values - x_critical)
    close_mask = distance < 0.1 * (np.max(x_values) - np.min(x_values))
    
    if np.sum(close_mask) < 5:
        return np.nan
    
    x_close = x_values[close_mask]
    obs_close = obs_data[close_mask]
    
    # Remove critical point itself
    non_critical = np.abs(x_close - x_critical) > 1e-10
    x_close = x_close[non_critical]
    obs_close = obs_close[non_critical]
    
    if len(x_close) < 3:
        return np.nan
    
    # Fit power law: obs ~ |x - x_c|^beta
    try:
        reduced_x = np.abs(x_close - x_critical)
        log_x = np.log(reduced_x)
        log_obs = np.log(np.abs(obs_close) + 1e-10)
        
        # Linear fit in log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_obs)
        
        if r_value**2 > 0.5:  # Reasonable fit
            return slope
        else:
            return np.nan
            
    except (ValueError, RuntimeWarning):
        return np.nan


def _apply_finite_size_scaling(result: PhaseTransitionResult, 
                              system_sizes: List[int]) -> PhaseTransitionResult:
    """Apply finite-size scaling corrections"""
    # Simplified implementation
    # In practice, would extrapolate to infinite size limit
    return result


def _bootstrap_analysis(result: PhaseTransitionResult, x_values: np.ndarray,
                       observables: Dict[str, np.ndarray], 
                       n_bootstrap: int) -> PhaseTransitionResult:
    """Perform bootstrap analysis for error estimation"""
    
    bootstrap_points = []
    
    for _ in range(min(n_bootstrap, 100)):  # Limit for performance
        # Resample data
        indices = np.random.choice(len(x_values), len(x_values), replace=True)
        x_boot = x_values[indices]
        obs_boot = {key: obs[indices] for key, obs in observables.items()}
        
        # Sort by x values
        sort_indices = np.argsort(x_boot)
        x_boot = x_boot[sort_indices]
        obs_boot = {key: obs[sort_indices] for key, obs in obs_boot.items()}
        
        try:
            # Quick detection on bootstrap sample
            boot_result = _derivative_detection(x_boot, obs_boot, 0.1)
            bootstrap_points.extend(boot_result.transition_points)
        except:
            continue
    
    # Update confidence scores based on bootstrap stability
    if bootstrap_points and result.transition_points:
        for i, point in enumerate(result.transition_points):
            # Count how many bootstrap samples found transition near this point
            tolerance = (np.max(x_values) - np.min(x_values)) * 0.05
            nearby_count = sum(1 for bp in bootstrap_points if abs(bp - point) < tolerance)
            stability = nearby_count / len(bootstrap_points) if bootstrap_points else 0
            
            # Update confidence
            if i < len(result.confidence_scores):
                result.confidence_scores[i] *= stability
    
    return result