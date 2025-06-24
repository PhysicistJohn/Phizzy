import pytest
import numpy as np
from phizzy.transitions import phase_transition_detect


class TestPhaseTransitionDetect:
    
    def test_simple_step_function(self):
        """Test detection of simple step transition"""
        # Create data with a clear discontinuous transition
        x = np.linspace(0, 10, 100)
        y = np.where(x < 5, 1.0, 3.0) + 0.1 * np.random.randn(100)
        
        result = phase_transition_detect(
            y, control_parameter=x, method='derivatives', 
            confidence_threshold=0.1
        )
        
        # Should detect transition near x=5
        assert len(result.transition_points) >= 1
        if result.transition_points:
            assert 4 < result.transition_points[0] < 6
    
    def test_continuous_transition(self):
        """Test detection of continuous transition"""
        # Create data resembling order parameter near critical point
        x = np.linspace(0, 2, 100)
        y = np.where(x > 1, np.sqrt(x - 1), 0) + 0.05 * np.random.randn(100)
        
        result = phase_transition_detect(
            y, control_parameter=x, method='clustering',
            confidence_threshold=0.1
        )
        
        assert isinstance(result.transition_points, list)
        assert isinstance(result.transition_types, list)
    
    def test_dictionary_input(self):
        """Test with dictionary input format"""
        x = np.linspace(0, 5, 50)
        magnetization = np.tanh(2 * (x - 2.5))
        energy = -x + 0.1 * np.random.randn(50)
        
        data = {
            'temperature': x,
            'magnetization': magnetization,
            'energy': energy
        }
        
        result = phase_transition_detect(
            data, control_parameter='temperature',
            confidence_threshold=0.1
        )
        
        assert hasattr(result, 'transition_points')
        assert hasattr(result, 'order_parameters')
    
    def test_ml_method(self):
        """Test machine learning detection method"""
        x = np.linspace(0, 4, 80)
        # Two-phase system
        y1 = np.where(x < 2, 1 + 0.1*x, 3 - 0.1*x) + 0.1 * np.random.randn(80)
        y2 = np.where(x < 2, 0.5, 1.5) + 0.1 * np.random.randn(80)
        
        observables = {'obs1': y1, 'obs2': y2}
        
        result = phase_transition_detect(
            y1, control_parameter=x, observables=observables,
            method='ml', confidence_threshold=0.1
        )
        
        assert isinstance(result.statistics, dict)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with missing control parameter in dict
        with pytest.raises(ValueError):
            phase_transition_detect(
                {'x': [1, 2, 3], 'y': [1, 2, 3]},
                control_parameter='missing_key'
            )
    
    def test_result_methods(self):
        """Test PhaseTransitionResult methods"""
        x = np.linspace(0, 10, 50)
        y = np.where(x < 5, 0, 1) + 0.1 * np.random.randn(50)
        
        result = phase_transition_detect(
            y, control_parameter=x, confidence_threshold=0.1
        )
        
        # Test get_phase_at method
        phase_info = result.get_phase_at(2.0)
        assert isinstance(phase_info, dict)