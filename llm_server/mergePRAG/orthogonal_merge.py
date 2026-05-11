import torch

def orthogonal_merging(WF: torch.Tensor | None, Wt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fuse slot matrices with the QR/Gram-Schmidt path used by the paper code.

    WF, Wt are shaped [num_kv, d_model]. The new memory is projected onto the
    orthogonal complement of the existing memory before being added, preserving
    the original slot count while reducing overwrite between passages.
    """
    if WF is None:
        return Wt
    if WF.shape != Wt.shape:
        raise ValueError(f"Cannot orthogonally merge tensors with different shapes: {WF.shape} vs {Wt.shape}")

    existing_cols = WF.transpose(0, 1).to(dtype=torch.float32)
    incoming_cols = Wt.transpose(0, 1).to(dtype=torch.float32)
    q_existing, _ = torch.linalg.qr(existing_cols, mode="reduced")
    projection = q_existing @ (q_existing.transpose(0, 1) @ incoming_cols)
    orthogonal_component = incoming_cols - projection
    fused = existing_cols + orthogonal_component
    return fused.transpose(0, 1).to(dtype=Wt.dtype)
