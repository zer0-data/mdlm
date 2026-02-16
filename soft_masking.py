import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMaskingModule(nn.Module):
    def __init__(self, hidden_size, vocab_size, mask_token_id, k=5, omega_s_init=-1.0, omega_a_init=0.0, omega_b_init=0.0):
        super().__init__()
        self.k = k
        self.mask_token_id = mask_token_id
        
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

    def forward(self, x_t, probs, embedding_layer):
        """
        x_t: (batch, seq_len) - Current token indices (with [MASK] tokens)
        probs: (batch, seq_len, vocab_size) - Predicted probabilities from Pass 1
        embedding_layer: function or module that takes indices and returns embeddings
        """
        # 1. Identify masks
        is_mask = (x_t == self.mask_token_id).unsqueeze(-1).float() # (batch, seq_len, 1)
        
        real_embeds = embedding_layer(x_t) # (batch, seq_len, hidden_dim)

        # Optimization: Only compute soft-masking if there are actually masked tokens
        if not is_mask.any():
            return real_embeds

        # 3. Get Mask Embedding
        mask_vector = embedding_layer(self.mask_token_id_tensor) # (hidden_dim)
        
        # 4. Get Feedback Embeddings & Lambda
        feedback_embeds = self.get_topk_embeddings(probs, embedding_layer) # (batch, seq_len, hidden_dim)
        lam = self.compute_lambda(probs) # (batch, seq_len, 1)
        
        # 5. Mix: Mask vs Feedback
        soft_mask_embeds = lam * feedback_embeds + (1 - lam) * mask_vector
        
        # 6. Final Combination using torch.where for better performance
        # Replaces embeddings only where is_mask is True
        final_embeds = torch.where(is_mask, soft_mask_embeds, real_embeds)
        
        return final_embeds
