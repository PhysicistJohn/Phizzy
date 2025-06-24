"""
Tests for experimental data interfaces.
"""

import pytest
import torch
import numpy as np
import tempfile
import h5py
import pandas as pd
from pathlib import Path
from phizzyML.data.experimental import HDF5DataInterface, CSVDataInterface, ExperimentalDataset
from phizzyML.data.preprocessing import DataNormalizer, OutlierDetector, PhysicsFeatureExtractor


class TestHDF5DataInterface:
    def test_hdf5_basic_loading(self):
        """Test basic HDF5 data loading."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        # Create test data
        test_data = {
            'energy': np.array([1.0, 2.0, 3.0, 4.0]),
            'momentum': np.array([0.5, 1.0, 1.5, 2.0])
        }
        
        with h5py.File(temp_path, 'w') as f:
            for key, data in test_data.items():
                dataset = f.create_dataset(key, data=data)
                dataset.attrs['units'] = 'J' if key == 'energy' else 'kg*m/s'
                dataset.attrs['uncertainty'] = '5%'
        
        # Load data
        loader = HDF5DataInterface(temp_path)
        loaded_data = loader.load()
        
        # Check data
        assert 'energy' in loaded_data
        assert 'momentum' in loaded_data
        assert torch.allclose(loaded_data['energy'], torch.tensor(test_data['energy'], dtype=torch.float32))
        
        # Check units
        assert loader.units['energy'].dimensions.length == 2  # J = kg⋅m²⋅s⁻²
        
        # Check uncertainties
        assert 'energy' in loader.uncertainties
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_hdf5_with_uncertainties(self):
        """Test loading data with uncertainties."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        # Create test data with uncertainties
        with h5py.File(temp_path, 'w') as f:
            dataset = f.create_dataset('measurement', data=np.array([10.0, 20.0, 30.0]))
            dataset.attrs['uncertainty'] = 1.5  # Absolute uncertainty
            dataset.attrs['systematic_uncertainty'] = 0.5
        
        loader = HDF5DataInterface(temp_path)
        uncertain_data = loader.get_with_uncertainties('measurement')
        
        assert uncertain_data.value.shape == (3,)
        assert uncertain_data.uncertainty.shape == (3,)
        assert torch.allclose(uncertain_data.uncertainty, torch.full((3,), 1.5))
        assert torch.allclose(uncertain_data.systematic, torch.full((3,), 0.5))
        
        Path(temp_path).unlink()


class TestCSVDataInterface:
    def test_csv_basic_loading(self):
        """Test basic CSV data loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
            
            # Write CSV with metadata
            f.write("# experiment: test_measurement\n")
            f.write("# units: mm\n")
            f.write("position,velocity,position_err,velocity_err\n")
            f.write("1.0,0.1,0.05,0.01\n")
            f.write("2.0,0.2,0.05,0.01\n")
            f.write("3.0,0.3,0.05,0.01\n")
        
        loader = CSVDataInterface(temp_path)
        data = loader.load()
        
        assert 'position' in data
        assert 'velocity' in data
        assert len(data['position']) == 3
        
        # Check uncertainties were loaded
        assert 'position' in loader.uncertainties
        assert 'velocity' in loader.uncertainties
        
        # Check metadata
        assert 'experiment' in loader.metadata
        
        Path(temp_path).unlink()


class TestDataNormalizer:
    def test_standard_normalization(self):
        """Test standard normalization."""
        data = {
            'x': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            'y': torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        
        normalizer = DataNormalizer(method='standard')
        normalizer.fit(data)
        normalized = normalizer.transform(data)
        
        # Check that normalized data has zero mean and unit variance
        assert torch.abs(normalized['x'].mean()) < 1e-6
        # Use unbiased=False to match sklearn's StandardScaler
        assert torch.abs(normalized['x'].std(unbiased=False) - 1.0) < 1e-6
        
        # Test inverse transform
        denormalized = normalizer.inverse_transform(normalized)
        assert torch.allclose(denormalized['x'], data['x'], atol=1e-6)
    
    def test_physics_normalization(self):
        """Test physics-aware normalization."""
        data = {
            'energy': torch.tensor([1.0, 4.0, 9.0, 16.0]),  # Squares
            'momentum': torch.tensor([0.1, 0.2, 0.3, 0.4])
        }
        
        normalizer = DataNormalizer(method='physics')
        normalizer.fit(data)
        normalized = normalizer.transform(data)
        
        # Check that normalization preserves physical meaning
        assert normalized['energy'].shape == data['energy'].shape
        assert normalized['momentum'].shape == data['momentum'].shape


class TestOutlierDetector:
    def test_physical_bounds_detection(self):
        """Test outlier detection with physical bounds."""
        # Create data with one obvious outlier
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 is outlier
        
        detector = OutlierDetector(method='physical_bounds')
        detector.fit(data, physical_bounds=(0, 10))
        
        mask = detector.predict(data)
        
        # Should identify the outlier
        assert mask[-1] == False  # Last value (100) should be outlier
        assert mask[:-1].all()    # First four should be inliers
    
    def test_statistical_outlier_detection(self):
        """Test statistical outlier detection."""
        # Generate normal data with outlier
        normal_data = torch.randn(100)
        outlier_data = torch.cat([normal_data, torch.tensor([10.0])])
        
        detector = OutlierDetector(method='statistical')
        detector.fit(outlier_data)
        
        mask = detector.predict(outlier_data)
        
        # Should detect the outlier
        assert mask[-1] == False  # Outlier should be detected
    
    def test_remove_outliers(self):
        """Test outlier removal from data dictionary."""
        data = {
            'x': torch.tensor([1.0, 2.0, 100.0, 4.0]),  # 100 is outlier
            'y': torch.tensor([2.0, 4.0, 200.0, 8.0])   # 200 is outlier
        }
        
        detector = OutlierDetector(method='physical_bounds')
        detector.fit(data['x'], physical_bounds=(0, 10))
        
        filtered = detector.remove_outliers(data, keys=['x'])
        
        # Should remove the outlier samples
        assert len(filtered['x']) == 3
        assert len(filtered['y']) == 3
        assert 100.0 not in filtered['x']


class TestPhysicsFeatureExtractor:
    def test_invariant_mass_calculation(self):
        """Test invariant mass feature extraction."""
        # Create 4-momentum data for particle at rest
        data = {
            'E': torch.tensor([10.0]),   # Energy
            'px': torch.tensor([0.0]),   # x-momentum
            'py': torch.tensor([0.0]),   # y-momentum
            'pz': torch.tensor([0.0])    # z-momentum
        }
        
        extractor = PhysicsFeatureExtractor()
        features = extractor.extract_features(data, ['invariant_mass'])
        
        # Invariant mass should equal energy for particle at rest
        assert 'invariant_mass' in features
        assert torch.allclose(features['invariant_mass'], data['E'])
    
    def test_transverse_momentum_calculation(self):
        """Test transverse momentum calculation."""
        data = {
            'px': torch.tensor([3.0]),
            'py': torch.tensor([4.0])
        }
        
        extractor = PhysicsFeatureExtractor()
        features = extractor.extract_features(data, ['transverse_momentum'])
        
        # Should be sqrt(3² + 4²) = 5
        assert 'transverse_momentum' in features
        assert torch.allclose(features['transverse_momentum'], torch.tensor([5.0]))
    
    def test_rapidity_calculation(self):
        """Test rapidity calculation."""
        data = {
            'E': torch.tensor([10.0]),
            'pz': torch.tensor([6.0])
        }
        
        extractor = PhysicsFeatureExtractor()
        features = extractor.extract_features(data, ['rapidity'])
        
        # Rapidity = 0.5 * ln((E + pz)/(E - pz))
        expected = 0.5 * torch.log(torch.tensor(16.0/4.0))
        assert 'rapidity' in features
        assert torch.allclose(features['rapidity'], expected)


class TestExperimentalDataset:
    def test_dataset_creation(self):
        """Test creating PyTorch dataset from experimental data."""
        # Create mock data loader
        class MockLoader:
            def __init__(self):
                self.uncertainties = {'input': torch.tensor([0.1, 0.1, 0.1])}
            
            def load(self):
                return {
                    'input': torch.tensor([1.0, 2.0, 3.0]),
                    'target': torch.tensor([2.0, 4.0, 6.0])
                }
        
        loader = MockLoader()
        dataset = ExperimentalDataset(
            loader, 
            input_keys=['input'], 
            target_keys=['target']
        )
        
        assert len(dataset) == 3
        
        # Test getting item
        sample = dataset[0]
        assert 'input' in sample
        assert 'target' in sample
        assert 'input_uncertainty' in sample
        
        assert sample['input'].item() == 1.0
        assert sample['target'].item() == 2.0