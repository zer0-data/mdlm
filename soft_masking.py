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

    def compute_lambda(self, probs):
        """
        Computes the mixing coefficient lambda.
        probs: (batch, seq_len, vocab_size) - Probability distribution
        Formula from paper Eq. 2: lambda(p) = omega_s * sigmoid(omega_a * (-H(p) - omega_b))
        """
        # Calculate Entropy
        # H(p) = - sum p * log p
        # Add epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1) # (batch, seq_len)
        
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
        # omega_b (offset) - let's keep it free as learning can adjust it.
        real_omega_b = self.omega_b
        
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
        topk_probs_norm = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
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
        
        # 2. Get Mask Embedding
        # We can just get it from the embedding layer using the mask definition
        # Assuming embedding_layer supports passing a tensor of IDs
        mask_vector = embedding_layer(torch.tensor(self.mask_token_id, device=x_t.device)) # (hidden_dim)
        
        # 3. Get Feedback Embeddings (Top-K mix)
        feedback_embeds = self.get_topk_embeddings(probs, embedding_layer) # (batch, seq_len, hidden_dim)

        # 4. Compute Lambda
        lam = self.compute_lambda(probs) # (batch, seq_len, 1)
        
        # 5. Mix: Mask vs Feedback
        # mixture = lambda * feedback + (1 - lambda) * mask
        soft_mask_embeds = lam * feedback_embeds + (1 - lam) * mask_vector
        
        # 6. Final Combination:
        # If token is NOT blocked/masked, use its original embedding.
        # If token IS masked, use the soft mixture.
        # However, MDLM usually feeds x_t which has [MASK] tokens.
        # So we just replacing the embedding at masked positions.
        
        real_embeds = embedding_layer(x_t)
        
        final_embeds = (1 - is_mask) * real_embeds + is_mask * soft_mask_embeds
        
        return final_embeds
