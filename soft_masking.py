import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMaskingModule(nn.Module):
    def __init__(self, hidden_size, vocab_size, mask_token_id, k=3, omega_s_init=0.5, omega_a_init=0.9, omega_b_init=0.3, interp_mode = 'linear', feedback_mode='topk', tau_init=1.0):
        super().__init__()
        self.k = k
        self.mask_token_id = mask_token_id

        valid_modes = ['linear', 'spherical']
        if interp_mode not in valid_modes:
            raise ValueError(f"interp_mode must be one of {valid_modes}")
        self.interp_mode = interp_mode

        valid_feedback = ['topk', 'full']
        if feedback_mode not in valid_feedback:
            raise ValueError(f"feedback_mode must be one of {valid_feedback}")
        self.feedback_mode = feedback_mode
        
        # Learnable parameters for lambda calculation
        # Formula: lambda = sigmoid(omega_s * Entropy + omega_a * t + omega_b)
        self.omega_s = nn.Parameter(torch.tensor(omega_s_init))
        self.omega_a = nn.Parameter(torch.tensor(omega_a_init))
        self.omega_b = nn.Parameter(torch.tensor(omega_b_init))

        # Learnable softmax temperature for 'full' feedback mode.
        # tau = exp(log_tau) is always positive.  Stored in log-space
        # for unconstrained optimisation.
        if self.feedback_mode == 'full':
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
        
        # Optimization: Register mask_token_id as a buffer to avoid repeated tensor creation
        self.register_buffer('mask_token_id_tensor', torch.tensor(mask_token_id,dtype= torch.long))

        

    def compute_lambda(self, probs):

        """
        Computes the mixing coefficient lambda.
        probs: (batch, seq_len, vocab_size) - Probability distribution
        Formula from paper Eq. 2: lambda(p) = omega_s * sigmoid(omega_a * (-H(p) - omega_b))
        """
        # Guard against AMP half-precision: entr() on near-zero fp16/bf16
        # values can produce NaNs.  Force float32 for the entropy path.
        probs = probs.float()

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
        
        # Re-parameterization to enforce the paper's parameter constraints (Eq. 2)
        # omega_s in [0, 1]  ->  hard clamp preserves full gradient within
        #   the valid range and avoids the sigmoid saturation that makes it
        #   nearly impossible to reach omega_s ≈ 1.0 (paper Fig. 3b).
        real_omega_s = self.omega_s.clamp(0.0, 1.0)
        # omega_a >= 0       ->  softplus is always positive
        real_omega_a = F.softplus(self.omega_a)
        # omega_b <= 0       ->  negated softplus is always negative.
        real_omega_b = -F.softplus(self.omega_b)

        # NOTE: intentional double-negative below.
        # Paper formula:  omega_a * (-H(p) - omega_b),  where omega_b <= 0.
        # So -omega_b >= 0, meaning the bias *adds* to -H(p), shifting the
        # sigmoid rightward (toward more certainty before masking kicks in).
        # Example: -H(p) - (-5)  ==  -H(p) + 5.  Mathematically correct.
        inner = real_omega_a * (-entropy - real_omega_b)  # == omega_a*(-H(p) - omega_b)
        sig = torch.sigmoid(inner)
        lam = real_omega_s * sig
        
        return lam.unsqueeze(-1) # (batch, seq_len, 1)

    def get_topk_embeddings(self, probs, embedding_layer):
        """
        Get weighted average of top-k embeddings.
        """
        # Zero out the [MASK] token probability before top-k selection.
        # If the model is uncertain, [MASK] can appear in top-k, wasting a
        # slot by blending the mask embedding with itself.
        probs = probs.clone()
        probs[..., self.mask_token_id] = 0.0

        topk_probs, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Normalize top-k probs
        topk_probs_norm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Get embeddings: (batch, seq_len, k, hidden_dim)
        topk_embeds = embedding_layer(topk_indices)
        
        # Weighted sum: (batch, seq_len, hidden_dim)
        feedback_embeds = torch.sum(topk_embeds * topk_probs_norm.unsqueeze(-1), dim=2)
        
        return feedback_embeds

    def get_full_vocab_embeddings(self, probs, embedding_layer):
        """Full-vocabulary temperature-scaled weighted embedding sum.

        Ablation variant (paper Table 3): instead of top-k filtering,
        use softmax(log(probs) / tau) over the *entire* vocabulary.
        tau is a learnable temperature.

        Returns: (batch, seq_len, hidden_dim)
        """
        tau = self.log_tau.exp()  # always positive
        # Temperature-scaled distribution over full vocabulary
        log_probs = torch.log(probs + 1e-10)
        scaled_probs = F.softmax(log_probs / tau, dim=-1)  # (B, L, V)
        # Matrix multiply: (B, L, V) @ (V, H) -> (B, L, H)
        embed_weight = embedding_layer.weight  # (V, H)
        return torch.matmul(scaled_probs, embed_weight)
    
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
        is_mask = (x_t == self.mask_token_id)          # (batch, seq_len) bool
        
        # 3. Base Embeddings (for unmasked positions these are the real tokens)
        real_embeds = embedding_layer(x_t)             # (batch, seq_len, H)
        
        # Optimization: Return early if no masks are present
        if not is_mask.any():
            return real_embeds

        # 4. Build v0: ALWAYS the mask-token embedding (paper Eq. 4)
        #    This avoids using real_embeds as v0, which would contain
        #    ground-truth embeddings at unmasked positions.
        mask_embed = embedding_layer(self.mask_token_id_tensor)  # (H,)
        v0 = mask_embed.expand_as(real_embeds)                   # (B, L, H)

        # 5. Compute feedback & lambda
        if self.feedback_mode == 'full':
            feedback_embeds = self.get_full_vocab_embeddings(probs, embedding_layer)
        else:
            feedback_embeds = self.get_topk_embeddings(probs, embedding_layer)
        lam = self.compute_lambda(probs)               # (B, L, 1)
        
        # 6. Interpolate: v0 = mask embedding, v1 = top-k feedback
        soft_mask_embeds = self._interpolate(lam, v0=v0, v1=feedback_embeds)
        
        # 7. Apply only at masked positions; unmasked keep real_embeds
        is_mask_3d = is_mask.unsqueeze(-1)             # (B, L, 1) bool
        final_embeds = torch.where(is_mask_3d, soft_mask_embeds, real_embeds)
        
        return final_embeds