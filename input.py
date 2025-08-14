from ase.io import read
from sella_interface import *

# Load your cluster model
xyz = 'R18_TS-fromSCAN.xyz'   # Change to your own .xyz file
atoms = read(xyz)

charge = 0         # Change based on your system charge
multiplicity = 1   # Change based on your system multiplicity

# Define atoms to fix
FREEZE_ATOMS = [1,9,10,19,23,24,33,44,45,55,58,59,64,66,67,72,83,84,94,103,104,                 # List of atoms to fix (1-index)
                113,121,122,136,150,160,163,166,177,187,196,206,217,218,259,272,
                276,281,286,287,299,300,303]     
fixed_indices = [a-1 for a in FREEZE_ATOMS]                                                     # Change atom indices into 0-based. Avoid if already 0-based are provided

# Run optimization with trajectory-based Hessian
ts_atoms = optimize_transition_state(
    atoms=atoms,
    charge=charge,
    mult=multiplicity,
    nthreads=16,                        # Number of threads (cpu-cores) for MLatom 
    fixed_indices=fixed_indices,
    method='AIQM2',                     # MLatom method
    max_steps=300,
    hessian_recalc=50,                  # Recalc analytical hessian every N steps
    output_file='ts_optimization.log',  # Logfile containing the Sella output
    fmax=0.01                          # the lower the number, the most strict is the optimization (0.01 recomended)
)

# Save final optimized structure
ts_atoms.write('ts_final_optimized.xyz')
