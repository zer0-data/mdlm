import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMaskingModule(nn.Module):
    def __init__(self, hidden_size, vocab_size, mask_token_id, k=3, omega_s_init=0.5, omega_a_init=0.9, omega_b_init=0.3, interp_mode = 'linear'):
        super().__init__()
        self.k = k
        self.mask_token_id = mask_token_id

        valid_modes = ['linear', 'spherical']
        if interp_mode not in valid_modes:
            raise ValueError(f"interp_mode must be one of {valid_modes}")
        self.interp_mode = interp_mode
        
        # Learnable parameters for lambda calculation
        # Formula: lambda = sigmoid(omega_s * Entropy + omega_a * t + omega_b)
        self.omega_s = nn.Parameter(torch.tensor(omega_s_init))
        self.omega_a = nn.Parameter(torch.tensor(omega_a_init))
        self.omega_b = nn.Parameter(torch.tensor(omega_b_init))
        
        # Optimization: Register mask_token_id as a buffer to avoid repeated tensor creation
        self.register_buffer('mask_token_id_tensor', torch.tensor(mask_token_id,dtype= torch.long))

        

    def compute_lambda(self, probs):

        """
        Computes the mixing coefficient lambda.
        probs: (batch, seq_len, vocab_size) - Probability distribution
        Formula from paper Eq. 2: lambda(p) = omega_s * sigmoid(omega_a * (-H(p) - omega_b))
        """
        # Calculate Entropy
        # H(p) = - sum p * log p
        # Add epsilon to avoid log(0)

        # log_probs = torch.log(probs + 1e-10)
        # entropy = -torch.sum(probs * log_probs, dim=-1) # (batch, seq_len)
        entropy = torch.special.entr(probs).sum(dim=-1) # (batch, seq_len)
        
        # Compute lambda
        # omega_s * sigmoid(omega_a * (-entropy - omega_b))
        # Note: -H(p) is negative entropy (higher confidence -> lower entropy -> higher negative entropy)
         
        # We need to ensure shapes broadcast correctly. 
        # entropy is (batch, seq_len)
        # parameters are scalars.
        
        # Re-parameterization to enforce constraints
        # omega_s in [0, 1] -> sigmoid
        real_omega_s = torch.sigmoid(self.omega_s)
        # omega_a >= 0 -> softplus
        real_omega_a = F.softplus(self.omega_a)
        # omega_b <= 0 (negative softplus)
        real_omega_b = -F.softplus(self.omega_b)
        
        inner = real_omega_a * (-entropy - real_omega_b)
        sig = torch.sigmoid(inner)
        lam = real_omega_s * sig
        
        return lam.unsqueeze(-1) # (batch, seq_len, 1)

    def get_topk_embeddings(self, probs, embedding_layer):
        """
        Get weighted average of top-k embeddings.
        """
        topk_probs, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Normalize top-k probs
        topk_probs_norm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Get embeddings: (batch, seq_len, k, hidden_dim)
        topk_embeds = embedding_layer(topk_indices)
        
        # Weighted sum: (batch, seq_len, hidden_dim)
        feedback_embeds = torch.sum(topk_embeds * topk_probs_norm.unsqueeze(-1), dim=2)
        
        return feedback_embeds
    
    def _interpolate(self, lam, v0, v1, dot_threshold=0.9995):
        """
        Routes the tensors through the selected interpolation strategy.
        lam: (batch, seq_len, 1) - mixing coefficient
        v0: (batch, seq_len, hidden_dim) - base mask embedding
        v1: (batch, seq_len, hidden_dim) - feedback prediction embedding
        """
        if self.interp_mode == 'linear':
            # Standard LERP
            return (1.0 - lam) * v0 + lam * v1
            
        elif self.interp_mode == 'spherical':
            # SLERP with LERP fallback for collinear vectors
            v0_norm = F.normalize(v0, p=2, dim=-1)
            v1_norm = F.normalize(v1, p=2, dim=-1)
            
            dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
            lerp_mask = (dot > dot_threshold)
            
            dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
            omega = torch.acos(dot)
            so = torch.sin(omega)
            
            c0 = torch.sin((1.0 - lam) * omega) / so
            c1 = torch.sin(lam * omega) / so
            
            slerp_out = c0 * v0 + c1 * v1
            lerp_out = (1.0 - lam) * v0 + lam * v1
            
            return torch.where(lerp_mask, lerp_out, slerp_out)

    def forward(self, x_t, probs, embedding_layer):
        """
        x_t: (batch, seq_len) - Current token indices (with [MASK] tokens)
        probs: (batch, seq_len, vocab_size) - Predicted probabilities from Pass 1
        embedding_layer: function or module that takes indices and returns embeddings
        """

       # 1. Input Validation
        if not torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-3):
            raise ValueError("Input `probs` must sum to 1. Did you pass raw logits?")

        # 2. Identify Masks
        is_mask = (x_t == self.mask_token_id).unsqueeze(-1) # (batch, seq_len, 1) bool
        
        # 3. Base Embeddings
        real_embeds = embedding_layer(x_t)
        
        # Optimization: Return early if no masks are present
        if not is_mask.any():
            return real_embeds

        # 4. Compute Feedback and Lambda
        feedback_embeds = self.get_topk_embeddings(probs, embedding_layer)
        lam = self.compute_lambda(probs)
        
        # 5. Route through the selected interpolation strategy
        # Note: real_embeds acts as the mask_vector (v0) at masked positions
        soft_mask_embeds = self._interpolate(lam, v0=real_embeds, v1=feedback_embeds)
        
        # 6. Final Masking
        final_embeds = torch.where(is_mask, soft_mask_embeds, real_embeds)
        
        return final_embeds