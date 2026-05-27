import torch
import os
from typing import Optional

class PLMEmbedder:
    """
    Handles ESM-2 inference and caching for protein sequences.
    Defaults to the 1280-dim model (esm2_t33_650M_UR50D) as per the DynaMo spec.
    """
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", cache_dir: str = "./data/processed/plm_embeddings/"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Loading ESM-2 model: {model_name}...")
        # Load via torch hub; in offline environments, provide a local path
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move to GPU if available for faster preprocessing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def get_embedding(self, sequence: str, protein_id: str, force_recompute: bool = False) -> torch.Tensor:
        """
        Extracts per-residue embeddings. Caches to disk using protein_id.
        Returns tensor of shape (N, 1280).
        """
        cache_path = os.path.join(self.cache_dir, f"{protein_id}.pt")
        
        if not force_recompute and os.path.exists(cache_path):
            return torch.load(cache_path)
            
        # Prepare data for ESM
        data = [(protein_id, sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Extract features from the final layer
        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        # Remove start <cls> and end <eos> tokens to match sequence length N
        sequence_embedding = token_representations[0, 1 : len(sequence) + 1]
        
        # Move back to CPU for storage
        sequence_embedding = sequence_embedding.cpu()
        torch.save(sequence_embedding, cache_path)
        
        return sequence_embedding
