from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from h_gar.model.autoregressive.diffusion_loss import DiffLoss
from h_gar.model.autoregressive.diffusion_action_loss import DiffActLoss
import pdb
from h_gar.model.autoregressive.utils_bank import MemoryBankCompressor, RouterMemoryBankSoftCompressor
import time
def mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len).to(device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).to(device),
    ).bool()
    return masking

class FineFrameCrossAttention(nn.Module):
    def __init__(self, action_dim=10, embed_dim=768, num_fine_tokens=256, n_heads=8):
        super().__init__()
        # self.num_fine_tokens = num_fine_tokens
        self.embed_dim = embed_dim 
        self.expand_ratio = 2 # 从 512 -> 1024，乘以 2
        self.expanded_dim = embed_dim * self.expand_ratio

        self.action_proj = nn.Linear(action_dim, embed_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_fine_tokens, embed_dim))

        self.cross_attn_goal = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)
        self.cross_attn_action = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)

        self.expand_proj = nn.Linear(embed_dim, self.expanded_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, z_goal, nactions):
        ### 1024 = 4 * 256
        #####2x16x10 2x256x768  ->2x1024x768
        # B = z_goal.size(0)
        B, T_g, D = z_goal.shape
        assert D == self.embed_dim, "z_goal embedding dim mismatch"
        ### z_goal [2, 256, 768]

        action_latent = self.action_proj(nactions)  # [B, 16, 768]
        learnable_queries = self.query_embed.expand(B, -1, -1)  # [B, 256, 768]
        queries = torch.cat([z_goal, learnable_queries], dim=1) # [B, 512, 768]

        # Cross-attend to goal
        z1, _ = self.cross_attn_goal(queries, queries, queries) # [B, 512, 768]
        z1 = z1 + queries  # residual # [B, 512, 768]
        z1 = self.norm1(z1)

        # Cross-attend to action
        z2, _ = self.cross_attn_action(z1, action_latent, action_latent) # [B, 512, 768]
        z2 = z2 + z1  # residual
        z2 = self.norm2(z2)

        # Projection [B,512,D]->[B,512,(D*2)]                          ###2x256x768->2x256x(768x4)->2x(256x4)x768
        z_expand = self.expand_proj(z2)  # [B, 512, 2D]

        T_e = z_expand.shape[1]

        # reshape to [B, 1024, D]
        z_fine = z_expand.view(B, T_e * self.expand_ratio, D) # # [B, 1024, D]

        # output proj
        z_fine = self.output_proj(z_fine)
        z_fine = F.layer_norm(z_fine, z_fine.shape[-1:])

        return z_fine

class FineGrainedActionDecoder(nn.Module):
    def __init__(self, action_dim=16, embed_dim=768, n_frames=4, coarse_len=16,n_heads=8):
        super().__init__()

        self.naction_proj = nn.Linear(action_dim, embed_dim)
        self.history_proj = nn.Linear(action_dim, embed_dim)
        self.h = 16
        self.w = 16
        self.num_actions = 16
        self.num_frames = n_frames

        # coarse action ←→ history interaction
        self.cross_attn_history = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        # fine-grained obs ←→ enhanced coarse action
        self.cross_attn_fine = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        # LayerNorms
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.inverse_refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.interpolate_back = nn.Linear(self.num_actions, self.num_frames)
        self.fc_back = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4→8
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 8→16
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        )



    def forward(self, z_fine, nactions, history_bank=None):
        """
        z_fine: [B, 1024, 768]
        nactions: [B, 16, 10]
        history_bank: [B, 32, 10]
        Returns:
            final: [B, 1024, 768]
        """
        B, C = z_fine.shape[0], z_fine.shape[2]

        # Step 1: 映射 action & memory 到 embed_dim
        nactions_emb = self.naction_proj(nactions)          # [B, 16, 768]
        history_emb = self.history_proj(history_bank)       # [B, 32, 768]

        # Step 2: Cross Attention (nactions ← history)
        attn_output1, _ = self.cross_attn_history(nactions_emb, history_emb, history_emb)###2x16x768  # [B, 16, 768]
        nactions_enhanced = self.norm1(nactions_emb + attn_output1)

        # Step 3: Cross Attention (z_fine ← nactions_enhanced)
        
        attn_output2, _ = self.cross_attn_fine(nactions_enhanced, z_fine, z_fine)  # [B, 1024, 768]
        z_fine_enhanced = self.norm2(nactions_enhanced + attn_output2)#####2x16x768    ->2x1024x768

        # attn_output2, _ = self.cross_attn_fine(nactions_emb, z_fine, z_fine)  # [B, 1024, 768]
        # z_fine_enhanced = self.norm2(nactions_emb + attn_output2)#####2x16x768    ->2x1024x768
        
        z_fine_enhanced = self.output_proj(z_fine_enhanced)#####2x16x768
        z = self.inverse_refine(z_fine_enhanced)
        z = z.permute(0, 2, 1)  
        z = self.interpolate_back(z) 
        z = z.permute(0, 2, 1)
        # Convert to [B*T, z_channels]
        z = z.reshape(B * self.num_frames, C)  # [B*T, 768]
        z = self.fc_back(z)                   # [B*T, 768×4×4]
        z = z.view(B * self.num_frames, C, 4, 4)  # [B*T, 768, 4, 4]
        z = self.deconv(z)  # → [B*T, 768, 16, 16]

        # Rearrange to [B, T, 256, 768]
        z = rearrange(z, "(b t) c h w -> b (t h w) c", b=B, t=self.num_frames)  # [B, 1024, 768]

        return z



class MAR(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        diffloss_d=3,
        diffloss_w=1024,
        diffloss_act_d=3,
        diffloss_act_w=1024,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=False,
        predict_video=True,
        act_diff_training_steps=1000,
        act_diff_testing_steps="100",
        action_model_params={},
        **kwargs
    ):
        super().__init__()

        self.task_name = kwargs["task_name"]
        self.different_history_freq = kwargs["different_history_freq"]
        self.use_history_action = kwargs["use_history_action"]
        self.action_mask_ratio = kwargs["action_mask_ratio"]
        self.use_proprioception = kwargs["use_proprioception"]
        self.predict_wrist_img = kwargs["predict_wrist_img"]
        self.predict_proprioception = kwargs["predict_proprioception"]
        self.n_frames = 4

        # ========= VAE and patchify specifics =========
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.vae_embed_dim = vae_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.label_drop_prob = label_drop_prob

        # ========= Masked MAE =========
        # variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # ========= Projection =========
        # conditional frames
        self.z_proj_cond = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)

        # video frames
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)

        # wrist video frames
        if self.predict_wrist_img:
            self.z_proj_wrist = nn.Linear(
                self.token_embed_dim, encoder_embed_dim, bias=True
            )

        # action
        self.predict_action = action_model_params["predict_action"]
        act_dim = kwargs["shape_meta"]["action"]["shape"][0]

        self.action_proj_cond = nn.Linear(act_dim, encoder_embed_dim, bias=True)
        self.buffer_size_action = 64

        # ========= Fake Latent =========
        self.fake_latent_x = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.fake_action_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        if self.predict_wrist_img:
            self.fake_latent_wrist_x = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        if self.use_history_action:
            self.fake_latent_history_action = nn.Parameter(
                torch.zeros(1, encoder_embed_dim)
            )

        # ========= History Action =========
        if self.use_history_action:
            self.history_action_proj_cond = nn.Linear(
                act_dim, encoder_embed_dim, bias=True
            )

        # ========= Proprioception =========
        if self.use_proprioception:
            self.buffer_size_properception = 64
            if self.different_history_freq:
                self.buffer_size_properception = 64 * 4

            if self.task_name == "umi":
                self.proprioception_proj_cond = nn.Linear(
                    16, encoder_embed_dim, bias=True
                )
            elif "pusht" in self.task_name:
                self.proprioception_proj_cond = nn.Linear(
                    2, encoder_embed_dim, bias=True
                )
            else:
                self.proprioception_proj_cond = nn.Linear(
                    9, encoder_embed_dim, bias=True
                )

            self.proprioception_image_proj_cond = nn.Linear(
                self.token_embed_dim, encoder_embed_dim, bias=True
            )

        # ========= Language Embedding =========
        self.language_emb_model = kwargs["language_emb_model"]
        self.language_emb_model_type = 1

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
                self.text_proj_cond = nn.Linear(
                    512, encoder_embed_dim, bias=True
                )  # clip text embedding is 512
                self.buffer_size_text = 64
                self.text_pos_embed = nn.Parameter(
                    torch.zeros(1, self.buffer_size_text, encoder_embed_dim)
                )

        # ========= Projection =========
        if self.predict_wrist_img:
            proj_cond_x_dim_num = 4
            if self.use_proprioception:
                proj_cond_x_dim_num += 2
            if self.use_history_action:
                proj_cond_x_dim_num += 1
        else:
            proj_cond_x_dim_num = 3
            if self.use_proprioception:
                if (
                    self.task_name == "umi"
                    or "block_push" in self.task_name
                    or "pusht" in self.task_name
                ):
                    proj_cond_x_dim_num += 1
                else:
                    proj_cond_x_dim_num += 2
            if self.use_history_action:
                proj_cond_x_dim_num += 1

        self.proj_cond_x_layer = nn.Linear(
            proj_cond_x_dim_num * encoder_embed_dim, encoder_embed_dim, bias=True
        )

        # ========= Temporal and Spatial Position Embedding =========
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, encoder_embed_dim)
        )  # Temporal position embedding, 4 frames
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim)
        )  # Spatial position embedding

        # ========= Normalization =========
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # ========= Encoder Blocks =========
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ========= Decoder =========
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # ========= Decoder Position Embedding =========
        self.decoder_temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, decoder_embed_dim)
        )
        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        # ========= Decoder Text Position Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                self.decoder_text_pos_embed = nn.Parameter(
                    torch.zeros(1, self.buffer_size_text, decoder_embed_dim)
                )

        # ========= Decoder Blocks =========
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(decoder_depth)
            ]
        )

        # ========= Decoder Norm =========
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # ========= Diffusion Temporal and Spatial Embedding =========
        self.diffusion_temporal_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, decoder_embed_dim)
        )
        self.diffusion_spatial_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        #########FineFrameCrossAttention###############
        self.cross_attention_module = FineFrameCrossAttention(
            action_dim=act_dim,
            embed_dim=encoder_embed_dim,
            num_fine_tokens=256,
            n_heads=8
        )

        self.action_memory_bank = MemoryBankCompressor(max_length=32)

        self.fine_action_decoder = FineGrainedActionDecoder(
            action_dim=act_dim,
            embed_dim=encoder_embed_dim,
            n_heads=8
            )


        # ========= Initialize Weights =========
        self.initialize_weights()

        # ========= Video Diffusion Loss =========
        self.predict_video = predict_video
        if self.predict_video:
            # ========= Video Diffusion Loss =========
            self.diffloss = DiffLoss(
                target_channels=self.token_embed_dim,
                z_channels=decoder_embed_dim,
                width=diffloss_w,
                depth=diffloss_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                n_frames=self.n_frames,
                language_emb_model=self.language_emb_model,
                language_emb_model_type=self.language_emb_model_type,
            )
            # self.fine_videodiffloss = DiffLoss(
            #     target_channels=self.token_embed_dim,
            #     z_channels=decoder_embed_dim,
            #     width=diffloss_w,
            #     depth=diffloss_d,
            #     num_sampling_steps=num_sampling_steps,
            #     grad_checkpointing=grad_checkpointing,
            #     n_frames=self.n_frames,
            #     language_emb_model=self.language_emb_model,
            #     language_emb_model_type=self.language_emb_model_type,
            # )

            # ========= Wrist Video Diffusion Loss =========
            if self.predict_wrist_img:
                self.diffloss_wrist = DiffLoss(
                    target_channels=self.token_embed_dim,
                    z_channels=decoder_embed_dim,
                    width=diffloss_w,
                    depth=diffloss_d,
                    num_sampling_steps=num_sampling_steps,
                    grad_checkpointing=grad_checkpointing,
                    n_frames=self.n_frames,
                    language_emb_model=self.language_emb_model,
                    language_emb_model_type=self.language_emb_model_type,
                )

        # ========= Action Diffusion Loss =========
        if self.predict_action:
            self.diffactloss = DiffActLoss(
                target_channels=act_dim,
                z_channels=decoder_embed_dim,
                width=diffloss_act_w,
                depth=diffloss_act_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                n_frames=self.n_frames,
                act_model_type=action_model_params["act_model_type"],
                act_diff_training_steps=act_diff_training_steps,
                act_diff_testing_steps=act_diff_testing_steps,
                language_emb_model=self.language_emb_model,
                language_emb_model_type=self.language_emb_model_type,
            )


        
        # ========= Proprioception Diffusion Loss =========
        if self.predict_proprioception:
            if self.task_name == "umi":
                self.diffproploss = DiffActLoss(
                    target_channels=6,
                    z_channels=decoder_embed_dim,
                    width=diffloss_act_w,
                    depth=diffloss_act_d,
                    num_sampling_steps=num_sampling_steps,
                    grad_checkpointing=grad_checkpointing,
                    n_frames=self.n_frames,
                    act_model_type=action_model_params["act_model_type"],
                    act_diff_training_steps=act_diff_training_steps,
                    act_diff_testing_steps=act_diff_testing_steps,
                    language_emb_model=self.language_emb_model,
                    language_emb_model_type=self.language_emb_model_type,
                )
            elif self.task_name == 'toolhang':
                self.diffproploss = DiffActLoss(
                        target_channels=9,
                        z_channels=decoder_embed_dim,
                        width=diffloss_act_w,
                        depth=diffloss_act_d,
                        num_sampling_steps=num_sampling_steps,
                        grad_checkpointing=grad_checkpointing,
                        n_frames=self.n_frames,
                        act_model_type=action_model_params["act_model_type"],
                        act_diff_training_steps=act_diff_training_steps,
                        act_diff_testing_steps=act_diff_testing_steps,
                        language_emb_model=self.language_emb_model,
                        language_emb_model_type=self.language_emb_model_type,
                    )
            else:
                raise NotImplementedError
            

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.cross_attention_module.query_embed, std=0.02)
        torch.nn.init.normal_(self.fake_latent_x, std=0.02)
        torch.nn.init.normal_(self.fake_action_latent, std=0.02)

        if self.predict_wrist_img:
            torch.nn.init.normal_(self.fake_latent_wrist_x, std=0.02)

        if self.use_history_action:
            torch.nn.init.normal_(self.fake_latent_history_action, std=0.02)

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                torch.nn.init.normal_(self.fake_latent, std=0.02)

        torch.nn.init.normal_(self.temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.spatial_pos_embed, std=0.02)

        torch.nn.init.normal_(self.decoder_temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_spatial_pos_embed, std=0.02)

        torch.nn.init.normal_(self.diffusion_temporal_embed, std=0.02)
        torch.nn.init.normal_(self.diffusion_spatial_embed, std=0.02)

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                torch.nn.init.normal_(self.text_pos_embed, std=0.02)
                torch.nn.init.normal_(self.decoder_text_pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.device).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, t, seq_len, embed_dim = x.shape

        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, t, seq_len, device=x.device)
        # Create the spatial mask for one frame (t=1)
        spatial_mask = torch.zeros(bsz, seq_len, device=x.device)
        spatial_mask = torch.scatter(
            spatial_mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        # Expand the spatial mask to all frames in the video
        mask = spatial_mask.unsqueeze(1).expand(-1, t, -1)

        return mask

    def forward_mae_encoder(
        self,
        x,
        mask,
        cond,
        text_latents=None,
        history_nactions=None,
        nactions=None,
        task_mode=None,
        proprioception_input={},
    ):
        B, T, S, _ = x.size()
        mask = rearrange(mask, "b t s -> b (t s)")
        ######2x4x256####
        # ========= Mask Input =========
        if task_mode == "policy_model":
            cond = self.z_proj_cond(cond)
            cond = rearrange(cond, "b t s c -> b (t s) c")
            x = self.fake_latent_x.unsqueeze(1).expand(B, cond.size(1), -1)

            if self.predict_wrist_img:
                wrist_x = self.fake_latent_wrist_x.unsqueeze(1).expand(
                    B, cond.size(1), -1
                )

        elif task_mode == "inverse_model":
            x = self.z_proj(x)
            x = rearrange(x, "b t s c -> b (t s) c")

            if self.predict_wrist_img:
                wrist_x = self.z_proj_wrist(proprioception_input["pred_second_image_z"])
                wrist_x = rearrange(wrist_x, "b t s c -> b (t s) c")

            cond = self.fake_latent_x.unsqueeze(1).expand(B, x.size(1), -1)

        else:
            cond = self.z_proj_cond(cond)
            cond = rearrange(cond, "b t s c -> b (t s) c")

            x = self.z_proj(x)
            x = rearrange(x, "b t s c -> b (t s) c")
            fake_latent_expanded = self.fake_latent_x.unsqueeze(1).expand(
                B, x.size(1), -1
            )
            x[mask == 1] = fake_latent_expanded[mask == 1].to(x.dtype)

            if self.predict_wrist_img:
                wrist_x = self.z_proj_wrist(proprioception_input["pred_second_image_z"])
                wrist_x = rearrange(wrist_x, "b t s c -> b (t s) c")
                fake_wrist_latent_expanded = self.fake_latent_wrist_x.unsqueeze(
                    1
                ).expand(B, wrist_x.size(1), -1)
                wrist_x[mask == 1] = fake_wrist_latent_expanded[mask == 1].to(
                    wrist_x.dtype
                )

        embed_dim = cond.size(2)

        # ========= History Action =========
        if self.use_history_action:#####None
            if history_nactions is None:
                history_action_latents = self.fake_latent_history_action.unsqueeze(
                    0
                ).repeat(B, T * self.n_frames, 1)
            else:
                history_action_latents = self.history_action_proj_cond(history_nactions)

                if self.training:
                    history_action_mask = (
                        torch.rand(B, T * self.n_frames) > self.action_mask_ratio
                    ).int()
                    history_action_latents[history_action_mask == 1] = (
                        self.fake_latent_history_action.to(history_action_latents.dtype)
                    )

            history_action_latents_expand = history_action_latents.repeat_interleave(
                self.buffer_size_action, dim=1
            )

        # ========= Proprioception =========
        if self.use_proprioception:###None
            if self.task_name == "umi":
                proprioception_state_cond = torch.cat(
                    [
                        proprioception_input["robot0_eef_pos"],
                        proprioception_input["robot0_eef_rot_axis_angle"],
                        proprioception_input["robot0_gripper_width"],
                        proprioception_input["robot0_eef_rot_axis_angle_wrt_start"],
                    ],
                    dim=-1,
                )
                proprioception_state_cond = self.proprioception_proj_cond(
                    proprioception_state_cond.float()
                )
                proprioception_state_cond_expand = (
                    proprioception_state_cond.repeat_interleave(
                        self.buffer_size_properception, dim=1
                    )
                )
            else:
                proprioception_image_cond = self.proprioception_image_proj_cond(
                    proprioception_input["second_image_z"]
                )
                proprioception_image_cond = rearrange(
                    proprioception_image_cond, "b t s c -> b (t s) c"
                )

                proprioception_state_cond = torch.cat(
                    [
                        proprioception_input["robot0_eef_pos"],
                        proprioception_input["robot0_eef_quat"],
                        proprioception_input["robot0_gripper_qpos"],
                    ],
                    dim=-1,
                )
                proprioception_state_cond = self.proprioception_proj_cond(
                    proprioception_state_cond
                )
                proprioception_state_cond_expand = (
                    proprioception_state_cond.repeat_interleave(
                        self.buffer_size_properception, dim=1
                    )
                )

        # ========= Action =========
        if task_mode == "dynamic_model":
            action_latents = self.action_proj_cond(nactions)
        else:
            action_latents = self.fake_action_latent.unsqueeze(0).repeat(B, 16, 1)
        action_latents_expand = action_latents.repeat_interleave(
            self.buffer_size_action, dim=1
        )####2x1024x768

        # ========= Wrist Video =========
        if self.predict_wrist_img:
            parts = [x, wrist_x, cond]
            if self.use_history_action:
                parts.append(history_action_latents_expand)
            parts.append(action_latents_expand)
            if self.use_proprioception:
                parts.extend(
                    [proprioception_image_cond, proprioception_state_cond_expand]
                )
            x = torch.cat(parts, dim=-1)
        else:
            parts = [x, cond]
            if self.use_history_action:
                parts.append(history_action_latents_expand)
            parts.append(action_latents_expand)

            if self.use_proprioception:
                if self.task_name == "umi":
                    parts.append(proprioception_state_cond_expand)
                else:
                    parts.extend(
                        [proprioception_image_cond, proprioception_state_cond_expand]
                    )
            x = torch.cat(parts, dim=-1)

        # ========= Projection =========
        x = self.proj_cond_x_layer(x)
        #####2x1024x2304 (768x3)
        # ========= Position Embedding =========
        temporal_pos_embed_expanded = self.temporal_pos_embed.unsqueeze(2).expand(
            -1, -1, S, -1
        ) #######1x4x256x768
        spatial_pos_embed_expanded = self.spatial_pos_embed.unsqueeze(1).expand(
            -1, T, -1, -1
        ) #####1x4x256x768

        combined_pos_embed = (
            temporal_pos_embed_expanded + spatial_pos_embed_expanded
        ).reshape(-1, T * S, embed_dim)#####1x1024x768
        x = x + combined_pos_embed

        # ========= Language Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                text_latents = text_latents.unsqueeze(1).repeat(
                    1, self.buffer_size_text, 1
                )####2x64x768

                ## this is for cfg
                if self.training:
                    drop_latent_mask = torch.rand(B) < self.label_drop_prob
                    drop_latent_mask = (
                        drop_latent_mask.unsqueeze(-1).to(self.device).to(x.dtype)
                    )
                    drop_latent_mask = drop_latent_mask.unsqueeze(1).repeat(
                        1, self.buffer_size_text, 1
                    )
                    text_latents = (
                        drop_latent_mask
                        * self.fake_latent.unsqueeze(1).repeat(
                            1, self.buffer_size_text, 1
                        )
                        + (1 - drop_latent_mask) * text_latents
                    )

                text_latents = text_latents + self.text_pos_embed
                x = torch.cat([text_latents, x], dim=1)
                ######2x1088x768
        # ========= Normalization =========
        x = self.z_proj_ln(x)

        # ========= Transformer Encoder Blocks =========
        if self.grad_checkpointing and not torch.jit.is_scripting():###None
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):
        B, T, S = mask.size()
        mask = rearrange(mask, "b t s -> b (t s)")
        x = self.decoder_embed(x)
        _, _, embed_dim = x.shape

        # ========= Position Embedding =========
        decoder_temporal_pos_embed_expanded = self.decoder_temporal_pos_embed.unsqueeze(
            2
        ).expand(
            -1, -1, S, -1
        ) 
        decoder_spatial_pos_embed_expanded = self.decoder_spatial_pos_embed.unsqueeze(
            1
        ).expand(
            -1, T, -1, -1
        ) 
        decoder_combined_pos_embed = (
            decoder_temporal_pos_embed_expanded + decoder_spatial_pos_embed_expanded
        ).reshape(1, T * S, embed_dim)

        # ========= Language Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                combined_pos_embed = torch.cat(
                    [self.decoder_text_pos_embed, decoder_combined_pos_embed], dim=1
                )
            else:
                combined_pos_embed = decoder_combined_pos_embed
        else:
            combined_pos_embed = decoder_combined_pos_embed

        x = x + combined_pos_embed

        # ========= Transformer Decoder Blocks =========
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        # ========= Language Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                x = x[:, self.buffer_size_text :]

        # ========= Diffusion Position Embedding =========
        diffusion_temporal_pos_embed_expanded = self.diffusion_temporal_embed.unsqueeze(
            2
        ).expand(
            -1, -1, S, -1
        )
        diffusion_spatial_pos_embed_expanded = self.diffusion_spatial_embed.unsqueeze(
            1
        ).expand(
            -1, T, -1, -1
        )
        diffusion_combined_pos_embed = (
            diffusion_temporal_pos_embed_expanded + diffusion_spatial_pos_embed_expanded
        ).reshape(1, T * S, embed_dim)

        x = x + diffusion_combined_pos_embed

        return x

    def forward_loss(
        self,
        z,
        target,
        mask,
        nactions=None,
        task_mode=None,
        gt_wrist_latents=None,
        gt_properception=None,
        text_latents=None,
        fine_grained = 'True'
    ):
        if task_mode == "video_model" or task_mode == "dynamic_model":
            if self.predict_wrist_img:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents, 
                )
                video_loss_wrist = self.diffloss_wrist(
                    z=z, target=gt_wrist_latents, mask=mask, text_latents=text_latents
                )
                video_loss = video_loss + video_loss_wrist
            else:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents, fine_grained = fine_grained
                )

            act_loss = torch.tensor(0.0).to(self.device)
            loss = video_loss

        elif task_mode == "policy_model" or task_mode == "inverse_model":
            act_loss = self.diffactloss(
                z=z, target=nactions, task_mode=task_mode, text_latents=text_latents
            )
            video_loss = torch.tensor(0.0).to(self.device)
            loss = act_loss

        elif task_mode == "full_dynamic_model":
            if self.predict_wrist_img:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents
                )
                video_loss_wrist = self.diffloss_wrist(
                    z=z, target=gt_wrist_latents, mask=mask, text_latents=text_latents
                )
                video_loss = video_loss + video_loss_wrist
            else:
                video_loss = self.diffloss(
                    z=z, target=target, mask=mask, text_latents=text_latents, fine_grained = fine_grained
                )
            act_loss = self.diffactloss(
                z=z, target=nactions, task_mode=task_mode, text_latents=text_latents
            )
            loss = video_loss + act_loss

        if self.predict_proprioception:
            properception_loss = self.diffproploss(
                z=z, target=gt_properception, text_latents=text_latents
            )
            loss = loss + properception_loss

        return loss, video_loss, act_loss

    def forward(
        self,
        imgs,
        cond,
        goal_image,
        history_nactions=None,
        nactions=None,
        text_latents=None,
        task_mode=None,
        proprioception_input={},
    ):
        self.device = cond.device
        B, T, C, H, W = imgs.size()

        # ========= Patchify =========
        imgs = rearrange(
            imgs, "b t c h w -> (b t) c h w"
        )
        x = self.patchify(imgs)
        x = rearrange(x, "(b t) seq_len c -> b t seq_len c", b=B)
        ######x: 2x4x256x16
        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(
            cond, "(b t) seq_len c -> b t seq_len c", b=B
        )  #######2x4x256x16
        ###cond: 2x4x256x16
        goal_image = rearrange(goal_image, "b t c h w -> (b t) c h w")
        goal_image = self.patchify(goal_image)
        goal_image = rearrange(
            goal_image, "(b t) seq_len c -> b t seq_len c", b=B
        )  #######2x4x256x16
        ##goal_image: 2x1x256x16

        # ========= Proprioception =========
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        # ========= Predicted Wrist Image =========
        if self.predict_wrist_img:
            if "pred_second_image_z" in proprioception_input:
                proprioception_input["pred_second_image_z"] = rearrange(
                    proprioception_input["pred_second_image_z"],
                    "b t c h w -> (b t) c h w",
                )
                proprioception_input["pred_second_image_z"] = self.patchify(
                    proprioception_input["pred_second_image_z"]
                )
                proprioception_input["pred_second_image_z"] = rearrange(
                    proprioception_input["pred_second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)
                #####2x768
        ######
        gt_goal_latents = goal_image.clone().detach()###2x1x256x16
        gt_latents = x.clone().detach()####2x4x256x16
        ########2x4x256x16
        # ========= Predicted Wrist Image =========
        if self.predict_wrist_img:
            if "pred_second_image_z" in proprioception_input:
                gt_wrist_latents = (
                    proprioception_input["pred_second_image_z"].clone().detach()
                )
                gt_wrist_latents = rearrange(
                    gt_wrist_latents, "b t s c -> b (t s) c"
                )

        # ========= Sample Orders =========
        orders = self.sample_orders(bsz=B)#####2x256
        mask = self.random_masking(x, orders)  # [1, 4, 256]
        mask_goal = mask[:, 0, :]

        # ========= MAE Encoder =========
        x = self.forward_mae_encoder(
            x,
            mask,
            cond,
            text_latents,
            history_nactions,
            nactions,
            task_mode=task_mode,
            proprioception_input=proprioception_input,
        )
        # ========= MAE Decoder =========
        z = self.forward_mae_decoder(x, mask)
                

        # ========= Diffloss over Video and Action =========
        mask = rearrange(mask, "b t s -> b (t s)")#######2x1024
        gt_latents = rearrange(
            gt_latents, "b t s c -> b (t s) c"     ######2x1024x16
        )

        gt_goal_latents = rearrange(
            gt_goal_latents, "b t s c -> b (t s) c"
        )
        z_goal = rearrange(
            z, "b (t s) c -> b t s c", t=T   ######2x1024x16    
        )[:, -1, :, :]
        
        #####2x1024x16
        # ========= Predict Proprioception =========
        if self.predict_proprioception:
            if self.task_name == "umi":
                gt_properception = proprioception_input[
                    "robot0_eef_rot_axis_angle_wrt_start_pred"
                ]
            elif self.task_name == "toolhang":
                gt_properception = torch.cat([proprioception_input['robot0_eef_pos_pred'], 
                                              proprioception_input['robot0_eef_quat_pred'], 
                                              proprioception_input['robot0_gripper_qpos_pred']], 
                                             dim=-1)
            else:
                raise NotImplementedError

            if self.predict_wrist_img:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_wrist_latents=gt_wrist_latents,
                    gt_properception=gt_properception,
                    text_latents=text_latents,
                )
            else:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_properception=gt_properception,
                    text_latents=text_latents,
                )
        else:
            if self.predict_wrist_img:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_wrist_latents=gt_wrist_latents,
                    text_latents=text_latents,
                )
            
            else:
                
                ######先生成目标帧 + corase_action##########################
                _, video_loss_coarse, _ = self.forward_loss(
                    z=z_goal,    #####2x256x768     2x256x1x768 ->2x16x10
                    target=gt_goal_latents,######2x256x16   
                    mask=mask_goal,#####2x256
                    nactions=nactions,####2x16x10       
                    task_mode="video_model",
                    text_latents=text_latents,###2x768
                    fine_grained = 'False'
                )
                _, _, act_loss_coarse = self.forward_loss(
                    z=z,    #####2x1024x768
                    target=gt_latents,######2x1024x768  
                    mask=mask,#####2x1024
                    nactions=nactions,####2x16x10       
                    task_mode="policy_model",
                    text_latents=text_latents,###2x768
                    fine_grained = 'False'
                )
                
                ####goal + coarse_act-> finegrained_frame######
                #########z->####2x1024x16->finegrained action#########

                z_fine = self.cross_attention_module(z_goal, nactions)####2x256x768 2x16x10  ->    2x256x768->2x256x768x4->2x1024x768
                    ######2x1024x768
                _, video_loss_fine, _ = self.forward_loss(
                    z=z_fine,    #####2x1024x768    2x4x256x768
                    target=gt_latents,######2x1024x16
                    mask=mask,#####2x1024
                    nactions=nactions,####2x16x10
                    task_mode="video_model",
                    # task_mode="full_dynamic_model",
                    text_latents=text_latents,###2x768
                    fine_grained = 'True'
                )
                for t in range(nactions.shape[1]):  # T=16
                    step_action = nactions[:, t]  # [B, 10]
                    history_bank = self.action_memory_bank.update(step_action)  #####2x16x10
                
                z_fine = self.fine_action_decoder(z_fine, nactions, history_bank)
                
                _, _, act_loss_fine_act = self.forward_loss(
                    z=z_fine,#####2x4x256x768
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode="policy_model",
                    text_latents=text_latents,
                    fine_grained = 'False'
                )
                video_loss = video_loss_fine + video_loss_coarse
                act_loss = act_loss_coarse + act_loss_fine_act
                loss = video_loss + act_loss
                
        return loss, video_loss, act_loss

    def sample_tokens(
        self,
        bsz,
        cond,
        text_latents=None,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        temperature=1.0,
        progress=False,
        history_nactions=None,
        nactions=None,
        proprioception_input={},
        task_mode=None,
        vae_model=None,
        x=None,
    ):
        
        self.device = cond.device
        B, T, C, H, W = cond.size()
        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(
            cond, "(b t) seq_len c -> b t seq_len c", b=B
        )

        # ========= Proprioception =========
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)

        # ========= Mask =========
        if task_mode == "inverse_model":
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.patchify(x)
            tokens = rearrange(
                x, "(b t) seq_len c -> b t seq_len c", b=B
            )
            mask = torch.zeros(bsz, self.n_frames, self.seq_len).to(self.device)
        else:
            # init and sample generation orders
            tokens = torch.zeros(
                bsz, self.n_frames, self.seq_len, self.token_embed_dim
            ).to(self.device)#####4x4x256x16
            mask = torch.ones(bsz, self.n_frames, self.seq_len).to(self.device)###4x4x256
            if self.predict_wrist_img:
                proprioception_input["pred_second_image_z"] = torch.zeros(
                    bsz, self.n_frames, self.seq_len, self.token_embed_dim
                ).to(self.device)

        # ========= Sample Orders =========
        orders = self.sample_orders(bsz)#####4x256

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # t1 = time.time()
        # t5 = time.time()
        # ========= Predict Video =========
        if self.predict_video:#####迭代num_iter次生成
            for step in indices:
                cur_tokens = tokens.clone()

                if self.predict_wrist_img:
                    cur_wrist_tokens = proprioception_input[
                        "pred_second_image_z"
                    ].clone()
                
                x = self.forward_mae_encoder(
                    tokens,
                    mask,
                    cond,
                    text_latents,
                    history_nactions=history_nactions,
                    nactions=nactions,
                    task_mode=task_mode,
                    proprioception_input=proprioception_input,
                )
                z = self.forward_mae_decoder(x, mask)
                #####4x1024x768
                z_goal = rearrange(
                    z, "b (t s) c -> b t s c", t=T   ######2x1024x16    
                )[:, -1, :, :]
                # t3 = time.time()
                if self.predict_action:
                    act_cfg = 1.0
                    sampled_token_latent_act_coarse = self.diffactloss.sample(
                        z, temperature, cfg=act_cfg, text_latents=text_latents
                    )########4x16x10
                    if self.action_memory_bank.compression_size is None:
                        for t in range(sampled_token_latent_act_coarse.shape[1]):  # T=16
                            step_action = sampled_token_latent_act_coarse[:, t]  # [B, 10]
                            history_bank = self.action_memory_bank.update(step_action)
                    
                    else:
                        history_bank = self.action_memory_bank.memory_bank 
                    z_fine = self.cross_attention_module(z_goal, sampled_token_latent_act_coarse)
                    z_fine = self.fine_action_decoder(z_fine, sampled_token_latent_act_coarse, history_bank)
                    sampled_token_latent_act = self.diffactloss.sample(
                            z_fine, temperature, cfg=act_cfg, text_latents=text_latents
                        )########4x16x10
                    for t in range(sampled_token_latent_act.shape[1]):  # T=16
                        step_action_fine = sampled_token_latent_act[:, t]  # [B, 10]
                        history_bank = self.action_memory_bank.update(step_action_fine)
                    
                else:
                    sampled_token_latent_act = None
                # t2 = time.time()
                # ========= Predict action and return if task_mode is inverse_model or policy_model=========
                if task_mode == "inverse_model" or task_mode == "policy_model":
                    return None, sampled_token_latent_act

                # ========= Mask Ratio =========
                # mask ratio for the next round, following MaskGIT and MAGE.
                mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(
                    self.device
                )

                # take the first frame mask
                mask_ = mask[:, 0]

                # masks out at least one for the next iteration
                mask_len = torch.maximum(
                    torch.Tensor([1]).to(self.device),
                    torch.minimum(
                        torch.sum(mask_, dim=-1, keepdims=True) - 1, mask_len
                    ),
                )

                # get masking for next iteration and locations to be predicted in this iteration
                mask_next = mask_by_order(
                    mask_len[0], orders, bsz, self.seq_len, self.device
                )

                ## expand mask_next to all frames
                mask_next = mask_next.unsqueeze(1).expand(-1, T, -1)
                mask_next = rearrange(mask_next, "b t s -> b (t s)")
                mask = rearrange(mask, "b t s -> b (t s)")

                if step >= num_iter - 1:
                    mask_to_pred = mask[:bsz].bool()
                else:
                    mask_to_pred = torch.logical_xor(
                        mask[:bsz].bool(), mask_next.bool()
                    )
                mask = mask_next
                mask = rearrange(mask, "b (t s) -> b t s", t=self.n_frames)

                if not cfg == 1.0:
                    mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

                # sample token latents for this step
                z = z[mask_to_pred.nonzero(as_tuple=True)]
                # cfg schedule follow Muse
                if cfg_schedule == "linear":
                    cfg_iter = (
                        1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                    )
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                else:
                    raise NotImplementedError

                sampled_token_latent = self.diffloss.sample(
                    z, temperature, cfg_iter, text_latents=text_latents
                )

                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(
                        2, dim=0
                    )  # Remove null class samples
                    mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

                cur_tokens = rearrange(cur_tokens, "b t s c -> b (t s) c")
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
                cur_tokens = rearrange(
                    cur_tokens, "b (t s) c -> b t s c", t=self.n_frames
                )
                tokens = cur_tokens.clone()

                # ========= Predict Wrist Image =========
                if self.predict_wrist_img:
                    sampled_wrist_token_latent = self.diffloss_wrist.sample(
                        z_fine, temperature, cfg_iter, text_latents=text_latents
                    )

                    if not cfg == 1.0:
                        sampled_wrist_token_latent, _ = (
                            sampled_wrist_token_latent.chunk(2, dim=0)
                        )  # Remove null class samples

                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b t s c -> b (t s) c"
                    )
                    cur_wrist_tokens[mask_to_pred.nonzero(as_tuple=True)] = (
                        sampled_wrist_token_latent
                    )
                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b (t s) c -> b t s c", t=self.n_frames
                    )
                    proprioception_input["pred_second_image_z"] = (
                        cur_wrist_tokens.clone()
                    )

            # ========= Unpatchify =========
            tokens = rearrange(tokens, "b t s c -> (b t) s c")
            tokens = self.unpatchify(tokens)
            # tokens = rearrange(tokens, '(b t) c h w -> b t c h w', b=B)

            if self.predict_wrist_img:
                wrist_tokens = rearrange(
                    proprioception_input["pred_second_image_z"], "b t s c -> (b t) s c"
                )
                wrist_tokens = self.unpatchify(wrist_tokens)

        else:
            raise NotImplementedError
        # t4 = time.time()
        if self.predict_wrist_img:
            return wrist_tokens, sampled_token_latent_act
        else:
            return tokens, sampled_token_latent_act


def mar_tiny(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=3,
        encoder_num_heads=6,
        decoder_embed_dim=768,
        decoder_depth=3,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_small(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=768,
        decoder_depth=6,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
