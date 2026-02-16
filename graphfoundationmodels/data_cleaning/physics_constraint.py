"""
Physics-Informed Constraints for Data-Driven Digital Twins

This module implements physics-based validation and constraints that can be
applied during data preprocessing and model training. Addresses Reviewer Concern #1.

Key Features:
- Domain-specific physics validation
- Measurement filtering based on physical limits
- Integration with selective loss calculation
"""

import numpy as np
import torch
from typing import Dict, Callable, Optional, Tuple
from abc import ABC, abstractmethod


class PhysicsConstraint(ABC):
    """Base class for physics-informed constraints"""
    
    @abstractmethod
    def validate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Validate data against physics constraints
        
        Args:
            data: Dictionary of measurement arrays
            
        Returns:
            Boolean mask where True indicates valid data
        """
        pass
    
    @abstractmethod
    def get_theoretical_limit(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate theoretical limit based on physics
        
        Args:
            data: Dictionary of input measurements
            
        Returns:
            Array of theoretical maximum/minimum values
        """
        pass


class PVPowerConstraint(PhysicsConstraint):
    """
    Photovoltaic power constraint: Power cannot exceed theoretical maximum
    
    Formula: max_power = irradiance * area * efficiency_limit
    Default efficiency_limit = 0.25 (25% maximum efficiency)
    
    References:
    - Shockley-Queisser limit for single-junction solar cells
    - Realistic field efficiency accounting for temperature, soiling, etc.
    """
    
    def __init__(self, area: float = 1.0, efficiency_limit: float = 0.25):
        """
        Args:
            area: Panel area in m^2
            efficiency_limit: Maximum theoretical efficiency (default 0.25 = 25%)
        """
        self.area = area
        self.efficiency_limit = efficiency_limit
    
    def validate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Validate PV power measurements
        
        Args:
            data: Must contain 'irradiance' (W/m^2) and 'dc_power' (W)
            
        Returns:
            Boolean mask: True where power ≤ theoretical maximum
        """
        irradiance = data['irradiance']
        dc_power = data['dc_power']
        
        max_power = self.get_theoretical_limit(data)
        
        # Allow small tolerance for measurement uncertainty
        tolerance = 1.05  # 5% tolerance
        valid_mask = dc_power <= (max_power * tolerance)
        
        # Also check for negative power (non-physical)
        valid_mask &= (dc_power >= 0)
        
        return valid_mask
    
    def get_theoretical_limit(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate theoretical maximum power
        
        Formula: P_max = G * A * η_max
        where G = irradiance, A = area, η_max = efficiency limit
        """
        irradiance = data['irradiance']
        return irradiance * self.area * self.efficiency_limit
    
    def flag_violations(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Identify and quantify constraint violations
        
        Returns:
            violation_mask: Boolean array of violations
            violation_rate: Fraction of measurements violating constraint
        """
        valid_mask = self.validate(data)
        violation_mask = ~valid_mask
        violation_rate = np.mean(violation_mask)
        
        return violation_mask, violation_rate


class LPBFEnergyDensityConstraint(PhysicsConstraint):
    """
    Laser Powder Bed Fusion energy density relationship
    
    Formula: energy_density = laser_power / scan_speed
    
    This physics-based relationship ensures that the reported energy density
    is consistent with the process parameters (laser power and scan speed).
    """
    
    def __init__(self, tolerance: float = 0.02):
        """
        Args:
            tolerance: Relative tolerance for energy density calculation (default 2%)
        """
        self.tolerance = tolerance
    
    def validate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Validate energy density calculations
        
        Args:
            data: Must contain 'laser_power' (W), 'scan_speed' (mm/s), 
                  and 'energy_density' (J/mm)
        """
        calculated_ed = self.get_theoretical_limit(data)
        reported_ed = data['energy_density']
        
        # Check relative error
        relative_error = np.abs(calculated_ed - reported_ed) / (calculated_ed + 1e-10)
        valid_mask = relative_error <= self.tolerance
        
        # Also check for non-physical values
        valid_mask &= (data['laser_power'] > 0)
        valid_mask &= (data['scan_speed'] > 0)
        valid_mask &= (data['energy_density'] > 0)
        
        return valid_mask
    
    def get_theoretical_limit(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate energy density from laser power and scan speed
        
        E = P / v
        where E = energy density, P = laser power, v = scan speed
        """
        laser_power = data['laser_power']  # W
        scan_speed = data['scan_speed']    # mm/s
        
        return laser_power / scan_speed    # J/mm


class DIWPositionConstraint(PhysicsConstraint):
    """
    Direct Ink Write position constraints based on machine limits
    
    Validates that position, velocity, and acceleration measurements
    are within physical machine limits.
    """
    
    def __init__(self, 
                 position_limits: Dict[str, Tuple[float, float]],
                 velocity_limit: float,
                 acceleration_limit: float):
        """
        Args:
            position_limits: Dict with 'x', 'y', 'z' keys, values are (min, max) tuples
            velocity_limit: Maximum velocity magnitude (mm/s)
            acceleration_limit: Maximum acceleration magnitude (mm/s^2)
        """
        self.position_limits = position_limits
        self.velocity_limit = velocity_limit
        self.acceleration_limit = acceleration_limit
    
    def validate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Validate DIW mechatronics measurements
        
        Args:
            data: Must contain position, velocity, acceleration for x, y, z axes
        """
        valid_mask = np.ones(len(data['x_pos']), dtype=bool)
        
        # Check position limits
        for axis in ['x', 'y', 'z']:
            pos = data[f'{axis}_pos']
            min_pos, max_pos = self.position_limits[axis]
            valid_mask &= (pos >= min_pos) & (pos <= max_pos)
        
        # Check velocity magnitude
        velocity_mag = np.sqrt(
            data['x_vel']**2 + data['y_vel']**2 + data['z_vel']**2
        )
        valid_mask &= (velocity_mag <= self.velocity_limit)
        
        # Check acceleration magnitude  
        accel_mag = np.sqrt(
            data['x_accel']**2 + data['y_accel']**2 + data['z_accel']**2
        )
        valid_mask &= (accel_mag <= self.acceleration_limit)
        
        return valid_mask
    
    def get_theoretical_limit(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Return the configured limits as a dictionary"""
        return {
            'position': self.position_limits,
            'velocity': self.velocity_limit,
            'acceleration': self.acceleration_limit
        }


class SelectiveLossCalculator:
    """
    Calculate loss only on validated/real measurements
    
    This addresses the concern that training on imputed values can bias
    the model. By using a mask to compute loss only on real measurements,
    we maintain fidelity to actual system behavior.
    
    Addresses Reviewer Concern #1 on physics-informed training.
    """
    
    def __init__(self, constraint: Optional[PhysicsConstraint] = None):
        """
        Args:
            constraint: Optional physics constraint to further filter training data
        """
        self.constraint = constraint
    
    def compute_loss(self,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     real_data_mask: torch.Tensor,
                     data_dict: Optional[Dict[str, np.ndarray]] = None,
                     criterion: Callable = torch.nn.functional.mse_loss) -> torch.Tensor:
        """
        Compute loss only on real (non-imputed) measurements that pass physics validation
        
        Args:
            predictions: Model predictions [batch, nodes, features, time]
            targets: Ground truth values [batch, nodes, features, time]
            real_data_mask: Boolean mask indicating real (non-imputed) measurements
            data_dict: Optional dictionary for physics validation
            criterion: Loss function (default MSE)
            
        Returns:
            Scalar loss computed only on valid measurements
        """
        # Start with real data mask
        valid_mask = real_data_mask.clone()
        
        # Apply physics constraints if provided
        if self.constraint is not None and data_dict is not None:
            physics_valid = self.constraint.validate(data_dict)
            physics_valid_tensor = torch.from_numpy(physics_valid).to(valid_mask.device)
            valid_mask = valid_mask & physics_valid_tensor
        
        # Select only valid measurements
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        # Compute loss
        if len(valid_predictions) > 0:
            loss = criterion(valid_predictions, valid_targets)
        else:
            # If no valid measurements, return zero loss
            loss = torch.tensor(0.0, device=predictions.device)
        
        return loss
    
    def get_validation_statistics(self, real_data_mask: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics on data validation
        
        Returns:
            Dictionary with validation statistics
        """
        total_measurements = real_data_mask.numel()
        valid_measurements = real_data_mask.sum().item()
        validation_rate = valid_measurements / total_measurements if total_measurements > 0 else 0.0
        
        return {
            'total_measurements': total_measurements,
            'valid_measurements': valid_measurements,
            'validation_rate': validation_rate,
            'rejected_measurements': total_measurements - valid_measurements
        }


# Example usage demonstrating physics-informed training
def example_pv_training():
    """
    Example: Training PV model with physics-informed selective loss
    """
    # Initialize physics constraint
    pv_constraint = PVPowerConstraint(area=50.0, efficiency_limit=0.25)
    
    # Initialize selective loss calculator
    loss_calculator = SelectiveLossCalculator(constraint=pv_constraint)
    
    # Simulated data
    batch_size, num_nodes, num_features, time_steps = 8, 29, 4, 96
    predictions = torch.randn(batch_size, num_nodes, num_features, time_steps)
    targets = torch.randn(batch_size, num_nodes, num_features, time_steps)
    real_data_mask = torch.rand(batch_size, num_nodes, num_features, time_steps) > 0.1  # 10% missing
    
    # Example data dict for physics validation
    data_dict = {
        'irradiance': np.random.rand(batch_size * num_nodes * time_steps) * 1000,
        'dc_power': np.random.rand(batch_size * num_nodes * time_steps) * 10000
    }
    
    # Compute selective loss
    loss = loss_calculator.compute_loss(
        predictions, 
        targets, 
        real_data_mask,
        data_dict
    )
    
    # Get validation statistics
    stats = loss_calculator.get_validation_statistics(real_data_mask)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Validation rate: {stats['validation_rate']:.2%}")
    print(f"Valid measurements: {stats['valid_measurements']:,} / {stats['total_measurements']:,}")


if __name__ == "__main__":
    print("Physics-Informed Constraints Module")
    print("=" * 50)
    print("\nThis module addresses Reviewer Concern #1:")
    print("Integration of physics-informed mechanisms in ddDT training")
    print("\nKey features:")
    print("1. Domain-specific physics validation (PV, L-PBF, DIW)")
    print("2. Selective loss calculation on validated measurements")
    print("3. Theoretical limit calculation based on physics")
    print("\nRunning example...")
    example_pv_training()