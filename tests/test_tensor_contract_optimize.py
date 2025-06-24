import pytest
import numpy as np
from phizzy.tensors import tensor_contract_optimize


class TestTensorContractOptimize:
    
    def test_matrix_multiplication(self):
        """Test basic matrix multiplication optimization"""
        A = np.random.randn(10, 20)
        B = np.random.randn(20, 15)
        
        result = tensor_contract_optimize('ij,jk->ik', [A, B])
        expected = A @ B
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
        assert result.result.shape == (10, 15)
        assert len(result.path) > 0
        assert result.cost >= 0
        assert result.scaling in ['linear', 'polynomial', 'exponential']
    
    def test_auto_equation_generation(self):
        """Test automatic equation generation from tensor shapes"""
        A = np.random.randn(5, 10)
        B = np.random.randn(10, 8)
        
        # Should automatically detect matrix multiplication
        result = tensor_contract_optimize([A, B])
        expected = A @ B
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
        assert result.result.shape == (5, 8)
    
    def test_tensor_chain_contraction(self):
        """Test chain of tensor contractions"""
        A = np.random.randn(4, 6)
        B = np.random.randn(6, 8)
        C = np.random.randn(8, 5)
        
        # Chain contraction: A·B·C
        result = tensor_contract_optimize('ij,jk,kl->il', [A, B, C])
        expected = A @ B @ C
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
        assert result.result.shape == (4, 5)
    
    def test_different_optimization_strategies(self):
        """Test different optimization strategies"""
        A = np.random.randn(10, 12)
        B = np.random.randn(12, 8)
        C = np.random.randn(8, 6)
        
        strategies = ['auto', 'greedy', 'optimal']
        results = []
        
        for strategy in strategies:
            result = tensor_contract_optimize(
                'ij,jk,kl->il', 
                [A, B, C], 
                optimize=strategy
            )
            results.append(result)
        
        # All should give same numerical result
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0].result, results[i].result, rtol=1e-10
            )
        
        # Greedy should be faster to optimize than optimal for this size
        assert results[1].optimization_time <= results[2].optimization_time
    
    def test_high_dimensional_tensors(self):
        """Test with higher-dimensional tensors"""
        A = np.random.randn(3, 4, 5)
        B = np.random.randn(5, 6, 7)
        
        result = tensor_contract_optimize('ijk,klm->ijlm', [A, B])
        
        # Check dimensions
        assert result.result.shape == (3, 4, 6, 7)
        
        # Manually verify contraction
        expected = np.zeros((3, 4, 6, 7))
        for i in range(3):
            for j in range(4):
                for l in range(6):
                    for m in range(7):
                        for k in range(5):
                            expected[i, j, l, m] += A[i, j, k] * B[k, l, m]
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
    
    def test_trace_operation(self):
        """Test trace-like operations"""
        A = np.random.randn(8, 8)
        
        # Trace
        result = tensor_contract_optimize('ii->', [A])
        expected = np.trace(A)
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
        assert result.result.shape == ()
    
    def test_outer_product(self):
        """Test outer product operations"""
        a = np.random.randn(5)
        b = np.random.randn(7)
        
        result = tensor_contract_optimize('i,j->ij', [a, b])
        expected = np.outer(a, b)
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
        assert result.result.shape == (5, 7)
    
    def test_precision_options(self):
        """Test different precision options"""
        A = np.random.randn(10, 10).astype(np.float32)
        B = np.random.randn(10, 10).astype(np.float32)
        
        # Test float32
        result32 = tensor_contract_optimize(
            'ij,jk->ik', [A, B], precision='float32'
        )
        assert result32.result.dtype == np.float32
        
        # Test float64
        result64 = tensor_contract_optimize(
            'ij,jk->ik', [A, B], precision='float64'
        )
        assert result64.result.dtype == np.float64
        
        # Test complex
        A_complex = A.astype(np.complex64)
        B_complex = B.astype(np.complex64)
        result_complex = tensor_contract_optimize(
            'ij,jk->ik', [A_complex, B_complex], precision='complex64'
        )
        assert result_complex.result.dtype == np.complex64
    
    def test_memory_limit(self):
        """Test memory limit constraints"""
        A = np.random.randn(20, 20)
        B = np.random.randn(20, 20)
        
        # Set very low memory limit
        result = tensor_contract_optimize(
            'ij,jk->ik', [A, B], 
            memory_limit=1000  # 1KB limit
        )
        
        # Should still work but might choose different path
        expected = A @ B
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
    
    def test_random_optimization(self):
        """Test random optimization strategy"""
        A = np.random.randn(8, 10)
        B = np.random.randn(10, 12)
        C = np.random.randn(12, 6)
        
        result = tensor_contract_optimize(
            'ij,jk,kl->il', [A, B, C], 
            optimize='random'
        )
        
        expected = A @ B @ C
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
    
    def test_simulated_annealing(self):
        """Test simulated annealing algorithm"""
        A = np.random.randn(6, 8)
        B = np.random.randn(8, 10)
        C = np.random.randn(10, 4)
        
        result = tensor_contract_optimize(
            'ij,jk,kl->il', [A, B, C], 
            algorithm='simulated-annealing'
        )
        
        expected = A @ B @ C
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
    
    def test_blas_options(self):
        """Test BLAS usage options"""
        A = np.random.randn(15, 20)
        B = np.random.randn(20, 25)
        
        # With BLAS
        result_blas = tensor_contract_optimize(
            'ij,jk->ik', [A, B], use_blas=True
        )
        
        # Without BLAS
        result_no_blas = tensor_contract_optimize(
            'ij,jk->ik', [A, B], use_blas=False
        )
        
        # Results should be the same
        np.testing.assert_allclose(
            result_blas.result, result_no_blas.result, rtol=1e-10
        )
    
    def test_performance_metrics(self):
        """Test that performance metrics are computed"""
        A = np.random.randn(10, 15)
        B = np.random.randn(15, 20)
        
        result = tensor_contract_optimize('ij,jk->ik', [A, B])
        
        # Check all metrics are present
        assert hasattr(result, 'optimization_time')
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'memory_usage')
        assert hasattr(result, 'cost')
        assert hasattr(result, 'scaling')
        
        # Check they have reasonable values
        assert result.optimization_time >= 0
        assert result.execution_time >= 0
        assert result.memory_usage > 0
        assert result.cost >= 0
        assert result.scaling in ['linear', 'polynomial', 'exponential']
    
    def test_complex_contraction_pattern(self):
        """Test complex contraction pattern"""
        # Simulate tensor network with multiple shared indices
        A = np.random.randn(4, 5, 6)
        B = np.random.randn(5, 7, 8)
        C = np.random.randn(6, 8, 9)
        
        # Contract over shared indices: sum over j and l
        result = tensor_contract_optimize('ijk,jlm,kln->imn', [A, B, C])
        
        # Manually compute expected result
        expected = np.zeros((4, 7, 9))
        for i in range(4):
            for m in range(7):
                for n in range(9):
                    for j in range(5):
                        for k in range(6):
                            for l in range(8):
                                expected[i, m, n] += A[i, j, k] * B[j, l, m] * C[k, l, n]
        
        np.testing.assert_allclose(result.result, expected, rtol=1e-10)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # No tensors provided
        with pytest.raises(ValueError):
            tensor_contract_optimize('ij->ij', [])
        
        # String equation without tensors
        with pytest.raises(ValueError):
            tensor_contract_optimize('ij,jk->ik')
        
        # Mismatched equation and tensors
        A = np.random.randn(5, 5)
        # This should still work as opt_einsum will handle it
        result = tensor_contract_optimize('ij,jk->ik', [A])
        assert result.result is not None
    
    def test_result_string_representation(self):
        """Test string representation of result"""
        A = np.random.randn(5, 8)
        B = np.random.randn(8, 10)
        
        result = tensor_contract_optimize('ij,jk->ik', [A, B])
        result_str = str(result)
        
        assert 'ContractionResult' in result_str
        assert 'shape=' in result_str
        assert 'cost=' in result_str
        assert 'scaling=' in result_str
    
    def test_vector_operations(self):
        """Test vector dot products and norms"""
        a = np.random.randn(100)
        b = np.random.randn(100)
        
        # Dot product
        result_dot = tensor_contract_optimize('i,i->', [a, b])
        expected_dot = np.dot(a, b)
        
        np.testing.assert_allclose(result_dot.result, expected_dot, rtol=1e-10)
        
        # Vector norm squared
        result_norm = tensor_contract_optimize('i,i->', [a, a])
        expected_norm = np.dot(a, a)
        
        np.testing.assert_allclose(result_norm.result, expected_norm, rtol=1e-10)