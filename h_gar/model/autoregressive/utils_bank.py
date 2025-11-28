import math

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterMemoryBankSoftCompressor(nn.Module):
    def __init__(self, task_name = 'libero10', action_dim=10, text_embed_dim=768, max_length=64, keep_ratio=0.75, hidden_dim=128):
        super().__init__()
        self.task_name = task_name
        self.max_length = max_length
        self.keep_ratio = keep_ratio
        self.keep = int(self.keep_ratio * self.max_length)
        self.memory_bank = None
        self.compression_size = None 

        # FiLM + router
        if self.task_name == 'libero10' or self.task_name == 'toolhang':
            self.scale = nn.Sequential(
                nn.Linear(text_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.shift = nn.Sequential(
                nn.Linear(text_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim)
            )
        self.router = nn.Linear(action_dim, 2)

        # FFN for selected tokens
        self.ffn = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, action_dim * 4),
            nn.GELU(),
            nn.Linear(action_dim * 4, action_dim)
        )

    def reset(self):
        self.memory_bank = None
        self.compression_size = None 

    def update(self, new_action: torch.Tensor, text_embed: torch.Tensor):
        """
        Add new action and compress if needed.
        new_action: [B, T, A]
        text_embed: [B, D_text]
        """
        B, T, A = new_action.shape
        if self.memory_bank is not None:
            self.memory_bank = self.memory_bank.detach()

        if self.memory_bank is None:
            self.memory_bank = new_action  # [B, T, A]
        else:
            self.memory_bank = torch.cat([self.memory_bank, new_action], dim=1)  # [B, T+1, A]

        if self.memory_bank.size(1) > self.max_length:
            self.memory_bank = self.compress(self.memory_bank, text_embed)

        return self.memory_bank

    def compress(self, memory_bank: torch.Tensor, text_embed: torch.Tensor,  new_token_count: int = 16):
        """
        Compress bank using router gating + FFN
        """
        B, T, A = memory_bank.shape
        keep_k = self.keep
        forced_keep = min(8, new_token_count)
        assert keep_k >= forced_keep
        # FiLM modulation
        if text_embed is not None:
            gamma = self.scale(text_embed).unsqueeze(1)  # [B, 1, A]2x1x10
            beta = self.shift(text_embed).unsqueeze(1)   # [B, 1, A]
            film_actions = memory_bank * (1 + gamma) + beta
        else:
            film_actions = memory_bank

        # Routing score
        logits = self.router(film_actions)  # [B, T, 2]


        keep_probs = F.softmax(logits, dim=-1)[:, :, 1]  # [B, T]


        new_token_start = T - new_token_count
        new_token_indices = torch.arange(new_token_start, T, device=memory_bank.device) 
        new_token_scores = keep_probs[:, new_token_start:T] 
        topk_new_local = torch.topk(new_token_scores, k=forced_keep, dim=1).indices 
        topk_new_global = torch.gather(
            new_token_indices.unsqueeze(0).expand(B, -1),  # [B, 16]
            dim=1,
            index=topk_new_local  # [B, K]
        )


        mask = torch.ones_like(keep_probs)
        mask[:, new_token_start:] = 0 
        masked_scores = keep_probs.masked_fill(mask == 0, float('-inf'))
        topk_old = torch.topk(masked_scores, k=keep_k - forced_keep, dim=1).indices  # [B, keep_k - forced_keep]
        topk_indices = torch.cat([topk_old, topk_new_global], dim=1)  # [B, keep_k]
        topk_indices = torch.sort(topk_indices, dim=1)[0]  

        # Top-K token selection
        # topk_indices = torch.topk(keep_probs, k=keep_k, dim=1).indices
        # topk_indices = torch.sort(topk_indices, dim=1)[0]

        # Gather top-k token & weights
        selected_tokens = torch.gather(
            memory_bank, dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, A)
        )
        selected_weights = torch.gather(keep_probs, dim=1, index=topk_indices)  # [B, K]

        # FFN + soft routing gate
        ffn_output = self.ffn(selected_tokens)  # [B, K, A]
        gated_output = ffn_output * selected_weights.unsqueeze(-1)  # [B, K, A]

        # Residual update
        updated_bank = selected_tokens + gated_output  # [B, K, A]
        return updated_bank


class MemoryBankCompressor:
    """
    Memory bank that stores nactions (coarse actions) and performs compression
    using cosine similarity and mean merging, based on MeMViT-style logic.
    """

    def __init__(self, max_length: int = 8):
        self.max_length = max_length
        self.memory_bank = None  # [B, T, A]
        self.compression_size = None  # [B, T, 1]

    def reset(self):
        self.memory_bank = None
        self.compression_size = None

    def update(self, new_action: torch.Tensor):
        """
        Add new action to memory and compress if needed.
        Args:
            new_action: [B, A]
        Returns:
            memory_bank: [B, <=max_length, A]
        """
        B, A = new_action.shape

        if self.memory_bank is None:
            self.memory_bank = new_action.unsqueeze(1)  # [B, 1, A]
            self.compression_size = torch.ones(B, 1, 1, device=new_action.device)  # [B, 1, 1]
        else:
            self.memory_bank = torch.cat([self.memory_bank, new_action.unsqueeze(1)], dim=1)  # [B, T+1, A]
            new_size = torch.ones(B, 1, 1, device=new_action.device)
            self.compression_size = torch.cat([self.compression_size, new_size], dim=1)  # [B, T+1, 1]

            if self.memory_bank.size(1) > self.max_length:
                self.memory_bank, self.compression_size = self._compress(self.memory_bank, self.compression_size)

        return self.memory_bank

    def _compress(self, memory_bank: torch.Tensor, compression_size: torch.Tensor):
        """
        Compress memory_bank using cosine similarity between adjacent time steps.
        Args:
            memory_bank: [B, T, A]
            compression_size: [B, T, 1]
        Returns:
            compressed_memory_bank: [B, T-1, A]
            compressed_size: [B, T-1, 1]
        """
        B, T, A = memory_bank.shape
        memory_bank = memory_bank.unsqueeze(2)  # [B, T, 1, A]2x9x1x10
        compression_size = compression_size.unsqueeze(2)  # [B, T, 1, 1]

        # cosine similarity between adjacent pairs
        similarity_matrix = F.cosine_similarity(memory_bank[:, :-1], memory_bank[:, 1:], dim=-1)  # [B, T-1, 1]

        _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)  # [B, 1, 1]

        # prepare indices
        src_indices = max_similarity_indices + 1  # to be merged into k  2x1x1             
        dst_indices = torch.arange(T - 1, device=memory_bank.device)[None, :, None].repeat(B, 1, 1)
        dst_indices[dst_indices >= max_similarity_indices.expand_as(dst_indices)] += 1

        # gather
        src_memory = memory_bank.gather(1, src_indices.unsqueeze(-1).expand(-1, -1, 1, A))
        dst_memory = memory_bank.gather(1, dst_indices.unsqueeze(-1).expand(-1, -1, 1, A))
        src_size = compression_size.gather(1, src_indices.unsqueeze(-1))
        dst_size = compression_size.gather(1, dst_indices.unsqueeze(-1))

        # weighted sum
        src_memory = src_memory * src_size
        dst_memory = dst_memory * dst_size

        # scatter add
        dst_memory.scatter_add_(1, max_similarity_indices.unsqueeze(-1).expand(-1, -1, 1, A), src_memory)
        dst_size.scatter_add_(1, max_similarity_indices.unsqueeze(-1), src_size)

        # normalize
        compressed_memory = dst_memory / dst_size
        compressed_memory = compressed_memory.squeeze(2)  # [B, T-1, A]
        dst_size = dst_size.squeeze(2)  # [B, T-1, 1]

        return compressed_memory, dst_size

