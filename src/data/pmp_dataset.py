import os
import torch
import pandas as pd
from typing import Callable, Optional
from torch_geometric.data import Dataset, Data

from src.data.graph_builder import ProteinGraphBuilder
from src.data.feature_extractor import ProteinFeatureExtractor
from src.data.plm_embedder import PLMEmbedder
from src.data.md_processor import MDProcessor

class PMPDataset(Dataset):
    """
    Peripheral Membrane Protein (PMP) Dataset.
    Integrates PLM embeddings, MD trajectories, and membrane geometry labels.
    """
    def __init__(self, root: str, split_file: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None):
        
        # Initialize sub-modules for processing
        self.graph_builder = ProteinGraphBuilder(k=16)
        self.feature_extractor = ProteinFeatureExtractor()
        self.plm_embedder = PLMEmbedder(cache_dir=os.path.join(root, "processed/plm_embeddings"))
        self.md_processor = MDProcessor(num_snapshots=10)
        
        # Read the split list (e.g., train.txt containing a list of PDB IDs)
        with open(split_file, 'r') as f:
            self.pdb_ids = [line.strip() for line in f.readlines()]
            
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # We expect a .pdb, a trajectory .xtc, and a labels .csv for every protein
        files = []
        for pid in self.pdb_ids:
            files.append(f"pdb/{pid}.pdb")
            files.append(f"md_trajectories/{pid}.xtc")
            files.append(f"labels/{pid}.csv")
        return files

    @property
    def processed_file_names(self):
        return [f"{pid}.pt" for pid in self.pdb_ids]

    def process(self):
        """
        Runs once to convert raw files -> PyG Data objects saved to disk.
        """
        for pid in self.pdb_ids:
            out_path = os.path.join(self.processed_dir, f"{pid}.pt")
            if os.path.exists(out_path):
                continue
                
            print(f"Processing {pid}...")
            
            # 1. Paths
            pdb_path = os.path.join(self.raw_dir, f"pdb/{pid}.pdb")
            traj_path = os.path.join(self.raw_dir, f"md_trajectories/{pid}.xtc")
            labels_path = os.path.join(self.raw_dir, f"labels/{pid}.csv")
            
            # 2. Extract MD data (snapshots, rmsf)
            md_data = self.md_processor.process_trajectory(pdb_path, traj_path)
            ref_coords = md_data['reference']
            snapshots = md_data['snapshots'] # (T, N, 3)
            rmsf = md_data['rmsf']           # (N, 1)
            
            # 3. Get PLM Embedding (Assumes sequence is extracted from PDB elsewhere)
            # For brevity, mocking sequence extraction. In reality, use BioPython here.
            sequence = "MKTLLL..." # Mock sequence
            plm_emb = self.plm_embedder.get_embedding(sequence, pid)
            
            # 4. Load Labels & Geometry (Mocked CSV read)
            # CSV should contain: binding_label, depth, tilt, sasa, amph_score, etc.
            df = pd.read_csv(labels_path)
            labels = torch.tensor(df['binding_label'].values, dtype=torch.float32)
            
            # 128 PLM + 47 Geom = 175. We extract the 47 geom features here
            geom_scalars = torch.tensor(df.iloc[:, 1:48].values, dtype=torch.float32) 
            # 4 specific membrane geom features for the right branch
            mem_geom_features = torch.tensor(df[['depth', 'tilt', 'mem_sasa', 'amph']].values, dtype=torch.float32)
            # 4 physicochemical features for the gate
            phys_features = torch.tensor(df[['kd', 'charge', 'rel_sasa', 'amph']].values, dtype=torch.float32)

            # 5. Extract structural features (s_node, v_node) for reference
            s_node_ref, v_node_ref = self.feature_extractor(ref_coords, plm_emb, geom_scalars)
            
            # 6. Build the graph based on reference coordinates
            data = self.graph_builder(ref_coords)
            
            # 7. Package everything into the PyG Data object
            data.s_node_ref = s_node_ref
            data.v_node_ref = v_node_ref
            data.snapshots = snapshots
            data.rmsf = rmsf
            data.mem_geom_features = mem_geom_features
            data.phys_features = phys_features
            data.y = labels
            
            # Note: During training (or here, if memory allows), you must also 
            # extract s_node_t and v_node_t for every snapshot in `data.snapshots`.
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, out_path)

    def len(self):
        return len(self.pdb_ids)

    def get(self, idx):
        pid = self.pdb_ids[idx]
        data = torch.load(os.path.join(self.processed_dir, f"{pid}.pt"))
        return data
