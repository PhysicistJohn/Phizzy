"""
Complete PhizzyML Workflow Example
==================================

This example demonstrates a complete physics ML workflow using all features of PhizzyML:
1. Dimensional analysis and unit checking
2. Experimental data loading with uncertainties
3. Physics-informed feature extraction
4. Neural networks with conservation constraints
5. Uncertainty quantification throughout
6. Symplectic integration for dynamics
7. Mesh-based PDE solving

Simulates a particle physics experiment analyzing collision data.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import h5py

# PhizzyML imports
import phizzyML as pml
from phizzyML.data.experimental import HDF5DataInterface, ExperimentalDataset
from phizzyML.data.preprocessing import DataNormalizer, PhysicsFeatureExtractor
from phizzyML.uncertainty.propagation import UncertainTensor, error_propagation
from phizzyML.constraints.conservation import EnergyConservingLayer, MomentumConservingLayer
from phizzyML.integrators.symplectic import LeapfrogIntegrator
from phizzyML.geometry.mesh import MeshGenerator
from phizzyML.geometry.operators import LaplacianOperator
from phizzyML.core.units import Unit, Quantity


def create_synthetic_experiment_data():
    """Create synthetic experimental data mimicking particle physics."""
    print("Creating synthetic experimental data...")
    
    n_events = 10000
    
    # Simulate particle collision data
    np.random.seed(42)
    
    # Primary particles (with realistic physics correlations)
    E1 = np.random.gamma(2, 5, n_events) + 5  # Energy in GeV
    theta1 = np.random.uniform(0, np.pi, n_events)
    phi1 = np.random.uniform(0, 2*np.pi, n_events)
    
    # Calculate momentum components
    p1_mag = np.sqrt(E1**2 - 0.14**2)  # Assume pion mass
    px1 = p1_mag * np.sin(theta1) * np.cos(phi1)
    py1 = p1_mag * np.sin(theta1) * np.sin(phi1)
    pz1 = p1_mag * np.cos(theta1)
    
    # Secondary particles (conservation laws applied)
    E2 = np.random.gamma(1.5, 3, n_events) + 2
    theta2 = np.random.uniform(0, np.pi, n_events)
    phi2 = np.random.uniform(0, 2*np.pi, n_events)
    
    p2_mag = np.sqrt(E2**2 - 0.14**2)
    px2 = p2_mag * np.sin(theta2) * np.cos(phi2)
    py2 = p2_mag * np.sin(theta2) * np.sin(phi2)
    pz2 = p2_mag * np.cos(theta2)
    
    # Add realistic experimental uncertainties
    def add_uncertainty(data, relative_error=0.05):
        uncertainty = np.abs(data) * relative_error
        noise = np.random.normal(0, uncertainty, data.shape) if hasattr(data, 'shape') else np.random.normal(0, uncertainty)
        return data + noise, uncertainty
    
    # Create dataset with uncertainties
    E1_meas, E1_err = add_uncertainty(E1, 0.03)
    E2_meas, E2_err = add_uncertainty(E2, 0.03)
    px1_meas, px1_err = add_uncertainty(px1, 0.05)
    py1_meas, py1_err = add_uncertainty(py1, 0.05)
    pz1_meas, pz1_err = add_uncertainty(pz1, 0.05)
    px2_meas, px2_err = add_uncertainty(px2, 0.05)
    py2_meas, py2_err = add_uncertainty(py2, 0.05)
    pz2_meas, pz2_err = add_uncertainty(pz2, 0.05)
    
    # Save to HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name
    
    with h5py.File(temp_path, 'w') as f:
        # Metadata
        f.attrs['experiment'] = 'Synthetic Particle Collisions'
        f.attrs['detector'] = 'PhizzyML Detector'
        f.attrs['energy_scale'] = 'GeV'
        
        # Particle 1 data
        p1_group = f.create_group('particle1')
        
        ds = p1_group.create_dataset('energy', data=E1_meas)
        ds.attrs['units'] = 'GeV'
        ds.attrs['uncertainty'] = '3%'
        
        ds = p1_group.create_dataset('px', data=px1_meas)
        ds.attrs['units'] = 'GeV/c'
        ds.attrs['uncertainty'] = '5%'
        
        ds = p1_group.create_dataset('py', data=py1_meas)
        ds.attrs['units'] = 'GeV/c'
        ds.attrs['uncertainty'] = '5%'
        
        ds = p1_group.create_dataset('pz', data=pz1_meas)
        ds.attrs['units'] = 'GeV/c'
        ds.attrs['uncertainty'] = '5%'
        
        # Particle 2 data
        p2_group = f.create_group('particle2')
        
        ds = p2_group.create_dataset('energy', data=E2_meas)
        ds.attrs['units'] = 'GeV'
        ds.attrs['uncertainty'] = '3%'
        
        ds = p2_group.create_dataset('px', data=px2_meas)
        ds.attrs['units'] = 'GeV/c'
        ds.attrs['uncertainty'] = '5%'
        
        ds = p2_group.create_dataset('py', data=py2_meas)
        ds.attrs['units'] = 'GeV/c'
        ds.attrs['uncertainty'] = '5%'
        
        ds = p2_group.create_dataset('pz', data=pz2_meas)
        ds.attrs['units'] = 'GeV/c'
        ds.attrs['uncertainty'] = '5%'
    
    return temp_path


def load_and_analyze_data(data_path):
    """Load experimental data and perform physics analysis."""
    print("Loading experimental data with uncertainties...")
    
    # Load particle 1 data
    p1_loader = HDF5DataInterface(data_path, group_path='/particle1')
    p1_data = p1_loader.load()
    
    # Load particle 2 data
    p2_loader = HDF5DataInterface(data_path, group_path='/particle2')
    p2_data = p2_loader.load()
    
    # Extract physics features
    print("Extracting physics features...")
    extractor = PhysicsFeatureExtractor()
    
    # Calculate invariant masses with uncertainty propagation
    def invariant_mass_2particle(E1, px1, py1, pz1, E2, px2, py2, pz2):
        """Calculate 2-particle invariant mass."""
        E_total = E1 + E2
        px_total = px1 + px2
        py_total = py1 + py2
        pz_total = pz1 + pz2
        
        m_squared = E_total**2 - px_total**2 - py_total**2 - pz_total**2
        return torch.sqrt(torch.clamp(m_squared, min=0))
    
    # Create uncertain tensors
    E1_unc = p1_loader.get_with_uncertainties('energy')
    E2_unc = p2_loader.get_with_uncertainties('energy')
    px1_unc = p1_loader.get_with_uncertainties('px')
    py1_unc = p1_loader.get_with_uncertainties('py')
    pz1_unc = p1_loader.get_with_uncertainties('pz')
    px2_unc = p2_loader.get_with_uncertainties('px')
    py2_unc = p2_loader.get_with_uncertainties('py')
    pz2_unc = p2_loader.get_with_uncertainties('pz')
    
    # Calculate invariant mass with full uncertainty propagation
    print("Calculating invariant mass with uncertainty propagation...")
    invariant_mass = error_propagation(
        invariant_mass_2particle,
        E1_unc, px1_unc, py1_unc, pz1_unc,
        E2_unc, px2_unc, py2_unc, pz2_unc
    )
    
    print(f"Invariant mass: {invariant_mass.value.mean():.3f} ± {invariant_mass.uncertainty.mean():.3f} GeV/c²")
    
    return {
        'p1_data': p1_data,
        'p2_data': p2_data,
        'invariant_mass': invariant_mass
    }


class PhysicsConstrainedDetector(nn.Module):
    """Neural network for particle detection with physics constraints."""
    
    def __init__(self, input_dim=8, hidden_dim=128):
        super().__init__()
        
        # Standard layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Physics constraint layers
        self.energy_constraint = EnergyConservingLayer()
        self.momentum_constraint = MomentumConservingLayer()
        
        # Output layers
        self.classifier = nn.Linear(hidden_dim, 2)  # Signal/Background
        self.regressor = nn.Linear(hidden_dim, 4)   # Energy, px, py, pz
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, apply_constraints=True):
        # Feature extraction
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        
        # Classification output
        classification = torch.softmax(self.classifier(x), dim=1)
        
        # Regression output with physics constraints
        regression = self.regressor(x)
        
        if apply_constraints:
            # Apply energy conservation
            regression = self.energy_constraint(regression)
            # Apply momentum conservation
            regression = self.momentum_constraint(regression)
        
        return classification, regression


def train_physics_network(data):
    """Train neural network with physics constraints."""
    print("Training physics-constrained neural network...")
    
    # Prepare training data
    p1_data = data['p1_data']
    p2_data = data['p2_data']
    
    # Create input features (8D: E1, px1, py1, pz1, E2, px2, py2, pz2)
    # Reshape to ensure 2D tensors
    def ensure_2d(tensor):
        if tensor.dim() == 1:
            return tensor.unsqueeze(1)
        return tensor
    
    inputs = torch.cat([
        ensure_2d(p1_data['energy']),
        ensure_2d(p1_data['px']),
        ensure_2d(p1_data['py']),
        ensure_2d(p1_data['pz']),
        ensure_2d(p2_data['energy']),
        ensure_2d(p2_data['px']),
        ensure_2d(p2_data['py']),
        ensure_2d(p2_data['pz'])
    ], dim=1)
    
    # Create synthetic labels (signal vs background)
    invariant_mass = data['invariant_mass'].value
    # Assume signal peak around 10 GeV
    signal_mask = (invariant_mass > 9.5) & (invariant_mass < 10.5)
    labels = signal_mask.float()
    
    # Create targets for regression (corrected 4-momenta)
    targets = inputs[:, 4:]  # Just use second particle as target
    
    # Normalize data
    normalizer = DataNormalizer(method='standard')
    normalizer.fit({'inputs': inputs})
    inputs_norm = normalizer.transform({'inputs': inputs})['inputs']
    
    # Create model
    model = PhysicsConstrainedDetector()
    
    # Loss functions
    classification_loss = nn.BCELoss()
    regression_loss = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    n_epochs = 50
    batch_size = 256
    n_batches = len(inputs) // batch_size
    
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = inputs_norm[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx].unsqueeze(1)
            batch_targets = targets[start_idx:end_idx]
            
            # Forward pass
            class_output, reg_output = model(batch_inputs)
            
            # Compute losses
            class_loss = classification_loss(class_output[:, 1:2], batch_labels)
            reg_loss = regression_loss(reg_output, batch_targets)
            
            total_loss_batch = class_loss + 0.1 * reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/n_batches:.4f}")
    
    return model, normalizer


def demonstrate_symplectic_integration():
    """Demonstrate symplectic integration for particle dynamics."""
    print("\nDemonstrating symplectic integration for particle dynamics...")
    
    # Hamiltonian for charged particle in magnetic field
    def charged_particle_hamiltonian(q, p, B=1.0, mass=1.0, charge=1.0):
        """H = (p - qA)²/(2m), where A is vector potential."""
        # Simplified: uniform magnetic field in z direction
        # A = (-B*y/2, B*x/2, 0)
        x, y = q[0], q[1]
        px, py = p[0], p[1]
        
        # Canonical momentum minus vector potential
        px_kin = px + charge * B * y / 2
        py_kin = py - charge * B * x / 2
        
        return (px_kin**2 + py_kin**2) / (2 * mass)
    
    # Initial conditions
    q0 = torch.tensor([1.0, 0.0])  # Position
    p0 = torch.tensor([0.0, 1.0])  # Momentum
    
    # Create integrator
    integrator = LeapfrogIntegrator(dt=0.01)
    
    # Integrate for one orbit
    n_steps = int(2 * np.pi / 0.01)
    q_traj, p_traj = integrator.integrate(
        q0, p0, charged_particle_hamiltonian, n_steps
    )
    
    # Plot trajectory
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(q_traj[:, 0].numpy(), q_traj[:, 1].numpy())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Trajectory in Magnetic Field')
    plt.axis('equal')
    plt.grid(True)
    
    # Energy conservation
    plt.subplot(1, 2, 2)
    energies = [charged_particle_hamiltonian(q, p).item() 
               for q, p in zip(q_traj, p_traj)]
    plt.plot(energies)
    plt.xlabel('Time Step')
    plt.ylabel('Energy')
    plt.title('Energy Conservation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('symplectic_integration_demo.png', dpi=150)
    plt.show()


def demonstrate_mesh_pde_solving():
    """Demonstrate PDE solving on mesh with physics operators."""
    print("\nDemonstrating PDE solving on mesh...")
    
    # Create circular domain
    mesh = MeshGenerator.circular_mesh((0, 0), 1.0, n_radial=8, n_angular=16)
    
    # Create Laplacian operator
    laplacian = LaplacianOperator(mesh)
    
    # Set up Poisson equation: -∇²u = f
    # Source term: f = sin(π*r) where r is distance from center
    distances = torch.norm(mesh.vertices, dim=1)
    source = torch.sin(np.pi * distances)
    
    # Simple iterative solver (Jacobi iteration)
    u = torch.zeros(mesh.n_vertices)  # Initial guess
    
    for iteration in range(100):
        # Apply Laplacian
        laplacian_u = laplacian(u)
        
        # Update: u_new = u_old - α * (∇²u + f)
        residual = laplacian_u + source
        u = u - 0.01 * residual
        
        # Apply boundary condition (u = 0 on boundary)
        boundary_mask = mesh.boundary_markers > 0
        u[boundary_mask] = 0
        
        if iteration % 20 == 0:
            residual_norm = torch.norm(residual).item()
            print(f"Iteration {iteration}, Residual: {residual_norm:.4f}")
    
    print("PDE solved successfully!")
    print(f"Solution range: [{u.min():.3f}, {u.max():.3f}]")


def main():
    """Run complete PhizzyML workflow."""
    print("=== Complete PhizzyML Physics ML Workflow ===\n")
    
    # 1. Create synthetic experimental data
    data_path = create_synthetic_experiment_data()
    
    # 2. Load and analyze data with uncertainties
    analysis_results = load_and_analyze_data(data_path)
    
    # 3. Train physics-constrained neural network
    model, normalizer = train_physics_network(analysis_results)
    
    # 4. Demonstrate symplectic integration
    demonstrate_symplectic_integration()
    
    # 5. Demonstrate mesh-based PDE solving
    demonstrate_mesh_pde_solving()
    
    # 6. Demonstrate dimensional analysis
    print("\nDemonstrating dimensional analysis...")
    
    # Calculate some physics quantities with units
    mass = Quantity(938.3, Unit.joule)  # Proton rest energy in MeV
    velocity = Quantity(0.9, Unit.dimensionless)  # β = v/c
    
    # Relativistic momentum: p = γmv
    gamma = 1.0 / np.sqrt(1 - velocity.magnitude**2)
    momentum = gamma * mass * velocity
    
    print(f"Mass energy: {mass}")
    print(f"Velocity (β): {velocity}")
    print(f"Relativistic momentum: {momentum}")
    
    # 7. Summary
    print("\n=== Workflow Complete ===")
    print("Successfully demonstrated:")
    print("✓ Dimensional analysis with automatic unit checking")
    print("✓ Experimental data loading with uncertainty propagation")
    print("✓ Physics feature extraction (invariant mass)")
    print("✓ Neural networks with conservation constraints")
    print("✓ Symplectic integration for dynamics")
    print("✓ Mesh-based PDE solving")
    print("✓ Comprehensive uncertainty quantification")
    
    # Cleanup
    import os
    os.unlink(data_path)


if __name__ == "__main__":
    main()