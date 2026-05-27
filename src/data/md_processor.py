import numpy as np
import torch
import MDAnalysis as mda
from MDAnalysis.analysis import rms

class MDProcessor:
    """
    Processes molecular dynamics trajectories to extract RMSF and conformational snapshots.
    """
    def __init__(self, num_snapshots: int = 10):
        self.num_snapshots = num_snapshots

    def process_trajectory(self, topology_path: str, trajectory_path: str) -> dict:
        """
        Args:
            topology_path: Path to .pdb or .tpr
            trajectory_path: Path to .xtc or .dcd
        Returns:
            Dictionary containing:
            - 'rmsf': torch.Tensor (N, 1)
            - 'snapshots': torch.Tensor (T, N, 3) 
            - 'reference': torch.Tensor (N, 3)
        """
        u = mda.Universe(topology_path, trajectory_path)
        calphas = u.select_atoms('protein and name CA')
        
        # 1. Calculate RMSF over the entire trajectory
        # Align trajectory to the first frame (reference) to compute RMSF accurately
        from MDAnalysis.analysis import align
        aligner = align.AlignTraj(u, u, select='protein and name CA', in_memory=True).run()
        
        rmsf_calc = rms.RMSF(calphas).run()
        rmsf_values = torch.tensor(rmsf_calc.rmsf, dtype=torch.float32).unsqueeze(-1)
        
        # 2. Extract T snapshots
        total_frames = len(u.trajectory)
        # Ensure we don't try to sample more frames than exist
        T = min(self.num_snapshots, total_frames)
        indices = np.linspace(0, total_frames - 1, T, dtype=int)
        
        snapshots = []
        for idx in indices:
            u.trajectory[idx]
            snapshots.append(torch.tensor(calphas.positions, dtype=torch.float32))
            
        snapshots_tensor = torch.stack(snapshots) # (T, N, 3)
        
        # 3. Extract reference coordinates (typically the 0th frame or the raw PDB)
        u.trajectory[0]
        reference_coords = torch.tensor(calphas.positions, dtype=torch.float32)
        
        return {
            'rmsf': rmsf_values,
            'snapshots': snapshots_tensor,
            'reference': reference_coords
        }
