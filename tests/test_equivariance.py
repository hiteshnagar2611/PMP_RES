import torch
import math
from src.data.feature_extractor import ProteinFeatureExtractor

def generate_random_rotation_matrix():
    """Generates a random 3x3 SO(3) rotation matrix using QR decomposition."""
    random_matrix = torch.randn(3, 3)
    q, r = torch.linalg.qr(random_matrix)
    # Ensure determinant is 1 (rotation, not reflection)
    d = torch.diag(r)
    ph = d.sign()
    q *= ph
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q

def test_so3_equivariance():
    """
    Tests that:
    1. Scalar features are invariant: f(R @ X) == f(X)
    2. Vector features are equivariant: f(R @ X) == R @ f(X)
    """
    # 1. Setup mock data
    N = 50 # Number of residues
    # Mock full backbone [N, CA, C] coordinates
    coords = torch.randn(N, 3, 3) 
    
    # Mock inputs
    plm_embeddings = torch.randn(N, 1280)
    # Mock geometric scalars (Dihedrals, SASA, depth, etc.) -> 47 dims
    geom_scalars = torch.randn(N, 47) 

    # 2. Initialize extractor (disable gradients & set to eval mode for deterministic output)
    extractor = ProteinFeatureExtractor()
    extractor.eval()
    
    with torch.no_grad():
        # 3. Base extraction
        s_base, V_base = extractor(coords, plm_embeddings, geom_scalars)
        
        # 4. Generate random rotation matrix and rotate coordinates
        R = generate_random_rotation_matrix()
        
        # Rotate coordinates: X_rot = X @ R^T
        # coords shape is (N, 3, 3) -> apply to the last dimension
        coords_rotated = torch.einsum('n a c, d c -> n a d', coords, R)
        
        # 5. Rotated extraction
        s_rot, V_rot = extractor(coords_rotated, plm_embeddings, geom_scalars)
        
        # 6. Assert Invariance for Scalars
        # s_base should exactly match s_rot
        scalar_diff = torch.max(torch.abs(s_base - s_rot)).item()
        assert torch.allclose(s_base, s_rot, atol=1e-5), \
            f"Scalar features are not invariant! Max diff: {scalar_diff}"
            
        # 7. Assert Equivariance for Vectors
        # Rotate the base vectors manually: V_expected = V_base @ R^T
        V_expected = torch.einsum('n v c, d c -> n v d', V_base, R)
        
        vector_diff = torch.max(torch.abs(V_expected - V_rot)).item()
        assert torch.allclose(V_expected, V_rot, atol=1e-5), \
            f"Vector features are not equivariant! Max diff: {vector_diff}"
            
        print("✅ Equivariance test passed successfully.")
        print(f"   Scalar invariance max error: {scalar_diff:.2e}")
        print(f"   Vector equivariance max error: {vector_diff:.2e}")

if __name__ == "__main__":
    test_so3_equivariance()
