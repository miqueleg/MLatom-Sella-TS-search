import numpy as np
import mlatom as ml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from sella import Sella, Constraints
from typing import List, Optional
import warnings
import os

class MLatomCalculator(Calculator):
    """ASE calculator wrapper for MLatom with trajectory saving"""
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, method='AIQM2', nthreads=8, charge=0, mult=1, 
                 trajectory_file='optimization_trajectory.xyz', **kwargs):
        Calculator.__init__(self, **kwargs)
        
        # Initialize MLatom model
        self.method = method
        self.charge = charge
        self.mult = mult
        self.nthreads = nthreads
        self.trajectory_file = trajectory_file
        self.step_counter = 0
        self.model = None
        self._initialize_model()
        
        # Initialize trajectory file
        self._initialize_trajectory_file()
    
    def _initialize_model(self):
        """Initialize the MLatom model"""
        try:
            if self.method.upper() == 'AIQM1':
                self.model = ml.models.methods(method='AIQM1', nthreads=self.nthreads)
            elif self.method.upper() == 'AIQM2':
                self.model = ml.models.methods(method='AIQM2', nthreads=self.nthreads)
            else:
                raise ValueError(f"Method {self.method} not supported")
                
            print(f"MLatom {self.method} model initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MLatom {self.method} model: {str(e)}")
    
    def _initialize_trajectory_file(self):
        """Initialize/clear the trajectory file"""
        if os.path.exists(self.trajectory_file):
            os.remove(self.trajectory_file)
        print(f"Trajectory will be saved to: {self.trajectory_file}")
    
    def _save_to_trajectory(self, atoms: Atoms):
        """Save current geometry to trajectory file"""
        try:
            # Create a copy to avoid modifying the original
            atoms_copy = atoms.copy()
            
            # Add step information as comment
            comment = f"Step {self.step_counter}, Energy: {self.results.get('energy', 'N/A')} eV"
            
            # Append to trajectory file
            from ase.io import write
            write(self.trajectory_file, atoms_copy, format='xyz', append=True, comment=comment)
            
            self.step_counter += 1
            
        except Exception as e:
            print(f"Warning: Could not save to trajectory file: {str(e)}")
    
    def _atoms_to_mlatom_molecule(self, atoms: Atoms) -> ml.data.molecule:
        """Convert ASE Atoms to MLatom molecule - using file approach"""
        # Create a temporary file for this calculation
        temp_xyz = f"temp_calc_{os.getpid()}.xyz"
        
        try:
            # Write current geometry to temporary file
            from ase.io import write
            write(temp_xyz, atoms, format='xyz')
            
            # Read with MLatom
            mol = ml.data.molecule.from_xyz_file(temp_xyz)
            
            # Set molecular properties
            mol.charge = self.charge
            mol.multiplicity = self.mult
            
            return mol
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_xyz):
                os.remove(temp_xyz)
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Main calculation method"""
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Convert to MLatom molecule
        mol = self._atoms_to_mlatom_molecule(atoms)
        molDB = ml.data.molecular_database([mol])
        
        # Compute energy and gradients
        try:
            self.model.predict(molecular_database=molDB, 
                             calculate_energy=True,
                             calculate_energy_gradients=True)
            
            # Extract results (MLatom gives energy in Hartree, gradients in Hartree/Angstrom)
            energy_hartree = molDB[0].energy
            forces_hartree_ang = -molDB[0].energy_gradients  # Forces = -gradients
            
            # Convert units: Hartree to eV, Hartree/Ang to eV/Ang
            hartree_to_ev = 27.211386245988
            
            self.results['energy'] = energy_hartree * hartree_to_ev
            self.results['forces'] = forces_hartree_ang * hartree_to_ev
            
            # Save current geometry to trajectory
            self._save_to_trajectory(atoms)
            
        except Exception as e:
            raise RuntimeError(f"MLatom calculation failed: {str(e)}")

def get_mlatom_hessian_from_trajectory(trajectory_file: str, charge=0, mult=1, 
                                     method='AIQM2', nthreads=8) -> np.ndarray:
    """
    External Hessian function for Sella using last structure from trajectory file
    
    Args:
        trajectory_file: Path to the optimization trajectory XYZ file
        charge: System charge
        mult: System multiplicity
        method: MLatom method to use
        nthreads: Number of threads
        
    Returns:
        Hessian matrix in eV/Ã…Â² units (3N x 3N)
    """
    
    # Initialize model if not cached
    if not hasattr(get_mlatom_hessian_from_trajectory, 'model') or \
       get_mlatom_hessian_from_trajectory.model is None:
        try:
            if method.upper() == 'AIQM1':
                get_mlatom_hessian_from_trajectory.model = ml.models.methods(method='AIQM1', nthreads=nthreads)
            elif method.upper() == 'AIQM2':
                get_mlatom_hessian_from_trajectory.model = ml.models.methods(method='AIQM2', nthreads=nthreads)
            else:
                raise ValueError(f"Method {method} not supported")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MLatom model: {str(e)}")
    
    # Read the last structure from trajectory file
    if not os.path.exists(trajectory_file):
        raise RuntimeError(f"Trajectory file {trajectory_file} not found")
    
    try:
        # Read all structures and get the last one
        from ase.io import read
        atoms_list = read(trajectory_file, index=':')  # Read all structures
        if not atoms_list:
            raise RuntimeError("No structures found in trajectory file")
        
        last_atoms = atoms_list[-1]  # Get the last structure
        print(f"Reading structure {len(atoms_list)} from trajectory for Hessian calculation")
        
        # Create temporary file for MLatom
        temp_xyz = f"temp_hess_{os.getpid()}.xyz"
        
        try:
            # Write last geometry to temporary file
            from ase.io import write
            write(temp_xyz, last_atoms, format='xyz')
            
            # Read with MLatom
            mol = ml.data.molecule.from_xyz_file(temp_xyz)
            
            # Set charge and multiplicity
            mol.charge = charge
            mol.multiplicity = mult
            
            molDB = ml.data.molecular_database([mol])
            
            # Calculate Hessian
            get_mlatom_hessian_from_trajectory.model.predict(molecular_database=molDB, 
                                                           calculate_hessian=True)
            
            # Extract Hessian (in Hartree/AngstromÂ²)
            hessian_hartree = molDB[0].hessian
            
            # Convert units: Hartree/AngÂ² to eV/AngÂ²
            hartree_to_ev = 27.211386245988
            hessian_ev = hessian_hartree * hartree_to_ev
            
            # Ensure symmetry
            hessian_ev = 0.5 * (hessian_ev + hessian_ev.T)
            
            return hessian_ev
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_xyz):
                os.remove(temp_xyz)
        
    except Exception as e:
        raise RuntimeError(f"MLatom Hessian calculation failed: {str(e)}")

def setup_cluster_constraints(atoms: Atoms, fixed_indices: List[int]) -> Atoms:
    """Setup constraints for fixing specific atoms"""
    
    if not fixed_indices:
        print("No atoms will be fixed")
        return atoms
    
    # Validate indices
    n_atoms = len(atoms)
    invalid_indices = [i for i in fixed_indices if i >= n_atoms or i < 0]
    if invalid_indices:
        raise ValueError(f"Invalid atom indices: {invalid_indices}. "
                        f"Must be in range 0-{n_atoms-1}")
    
    # Apply FixAtoms constraint
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    
    print(f"Fixed {len(fixed_indices)} atoms: {fixed_indices}")
    return atoms

def optimize_transition_state(atoms: Atoms,
                            fixed_indices: List[int], 
                            method: str = 'AIQM1',
                            nthreads=8,
                            charge=0,
                            mult=1,
                            hessian_recalc = 20,
                            trajectory_file: str = 'optimization_trajectory.xyz',
                            output_file: str = 'ts_optimization.log',
                            max_steps: int = 200,
                            fmax: float = 0.01) -> Atoms:
    """Main function to optimize transition state using Sella + MLatom with trajectory saving"""
    
    print("=" * 60)
    print("TRANSITION STATE OPTIMIZATION")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Initial atoms: {len(atoms)}")
    print(f"Fixed atoms: {len(fixed_indices)}")
    print(f"Max steps: {max_steps}")
    print(f"Force threshold: {fmax} eV/Ã…")
    print(f"Trajectory file: {trajectory_file}")
    print("=" * 60)
    
    # Setup constraints
    atoms = setup_cluster_constraints(atoms, fixed_indices)
    
    # Set up MLatom calculator with trajectory saving
    calc = MLatomCalculator(method=method, nthreads=nthreads, charge=charge, 
                           mult=mult, trajectory_file=trajectory_file)
    atoms.set_calculator(calc)
    
    # Create external Hessian function that reads from trajectory
    def hessian_function(atoms_current):
        print(f"Computing Hessian from latest structure in {trajectory_file}")
        return get_mlatom_hessian_from_trajectory(trajectory_file, charge=charge, 
                                                mult=mult, method=method, nthreads=nthreads)
    
    # Check initial forces
    print("Calculating initial forces...")
    initial_forces = atoms.get_forces()
    max_initial_force = np.max(np.abs(initial_forces))
    print(f"Initial max force: {max_initial_force:.6f} eV/Ã…")
    
    if max_initial_force < fmax:
        print("WARNING: System is already converged!")
        print("Computing initial Hessian to check transition state character...")
        hess = hessian_function(atoms)
        eigenvals = np.linalg.eigvals(hess)
        n_negative = np.sum(eigenvals < -1e-6)
        print(f"Number of negative eigenvalues: {n_negative}")
        
        if n_negative == 1:
            print("âœ“ This is a transition state (1 negative eigenvalue)")
        elif n_negative == 0:
            print("âš  This appears to be a minimum (0 negative eigenvalues)")
        else:
            print(f"âš  Higher-order saddle point ({n_negative} negative eigenvalues)")
    
    # Initialize Sella optimizer for transition state search
    try:
        opt = Sella(
            atoms,
            order=1,  # First-order saddle point (transition state)
            internal=True,  # Use internal coordinates for better convergence
            hessian_function=hessian_function,  # External Hessian from trajectory
            diag_every_n=hessian_recalc,  # Recalculate Hessian every 5 steps
            logfile=output_file
        )
        
        print("Sella optimizer initialized successfully")
        print("Starting transition state optimization...")
        
        # Run optimization
        opt.run(fmax=fmax, steps=max_steps)
        
        # Get final results
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        max_force = np.max(np.abs(final_forces))
        
        print("=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Final energy: {final_energy:.6f} eV")
        print(f"Max force: {max_force:.6f} eV/Ã…")
        print(f"Converged: {max_force < fmax}")
        print(f"Trajectory saved in: {trajectory_file}")
        print(f"Log file: {output_file}")
        
        return atoms
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise

