import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    # divide the dimension (dim) into frame (f), height (h), and width (w)
    # calculate the RoPE values for the frame (f) dimension
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta) # torch.Size([end, (dim - 2 * (dim // 3)) // 2])

    # calculate the RoPE values for the height (h) dimension
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta) # torch.Size([end, (dim // 3) // 2])

    # calculate the RoPE values for the width (w) dimension
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta) # torch.Size([end, (dim // 3) // 2])

    # all values in these tensors are complex numbers, and exist on the unit circle in the complex plane.
    # 이것을 풀어 설명하면 다음과 같습니다.`
    # the magnitude of each value is always 1.
    # the range of the real part (cos) is [-1, 1].
    # the range of the imaginary part (sin) is [-1, 1].
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # | theta value     | rotation frequency | rotation period (wavelength) | feature |    
    # | large (e.g., 10000) | low (slow rotation) | long             | useful for long-range dependency (standard) |
    # | small (e.g., 500) | high (fast rotation) | short            | useful for local dependency |    

    # 1d rope precompute
    # 1. calculate the rotation frequency to use in RoPE. the formula is 1 / (theta^(2k/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    # 2. outer product of each position (0~end-1) and frequency to create a matrix of shape (end, dim/2)
    #    this is the actual rotation angle (m * θ) at each position
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    # 3. convert the rotation angle to a complex number using the Euler formula (e^(i * mθ) = cos(mθ) + i*sin(mθ))
    #    this complex number is multiplied by the query/key vector later to perform the rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 # create a complex number with magnitude r and angle theta
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    # 1. reshape the real tensor x to (..., n, d)
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)

    # 2. convert the real tensor x to a complex tensor
    #    (..., d) to (..., d/2, 2) and treat it as a complex tensor
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))

    # 3. multiply the complex tensors to apply the rotation
    #    here freqs is exactly freqs_cis
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class LoRALinear(nn.Module):
    """
    manual LoRA implementation for nn.Linear layer.
    freeze the weights of the original layer, and only train LoRA matrices A, B.
    """
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.original_layer = original_layer
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.scale = alpha / rank
        
        # freeze the weights of the original layer
        self.original_layer.requires_grad_(False)

    def forward(self, x):
        # add the output of the original layer and the output of the LoRA layer
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scale
        return original_output + lora_output

# new layer!
class SelfAttentionPre(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, rank: int = 0, alpha: int = 16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim) # 1536, 1536
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.rank = rank

        if rank > 0:
            self.q_lora = LoRALinear(self.q, rank, alpha)
            self.k_lora = LoRALinear(self.k, rank, alpha)
            self.v_lora = LoRALinear(self.v, rank, alpha)

            self.norm_q_lora = RMSNorm(dim, eps=eps)
            self.norm_k_lora = RMSNorm(dim, eps=eps)            

        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        batch_size, frame_size, height, width, dim = x.shape

        # x: (b_half f) three h w d
        x_trg = x[:, 0:1, ...]
        x_src = x[:, 1:, ...]
        
        # x_trg = rearrange(x_trg, 'fb one h w d -> fb (one h w) d', one=1) # b, s, d
        # x_src = rearrange(x_src, 'fb two h w d -> fb (two h w) d', two=2) # b, s, d

        # source
        if self.rank > 0:
            q_src = self.norm_q_lora(self.q_lora(x_src))
            k_src = self.norm_k_lora(self.k_lora(x_src))
            v_src = self.v_lora(x_src)
        else:
            q_src = self.norm_q(self.q(x_src))
            k_src = self.norm_k(self.k(x_src))
            v_src = self.v(x_src)

        # target
        q = self.norm_q(self.q(x_trg))
        k = self.norm_k(self.k(x_trg))
        v = self.v(x_trg)

        # reunion        
        q = torch.cat([q, q_src], dim=1)
        q = rearrange(q, 'fb three h w d -> fb (three h w) d', three=3) # b, s, d
        
        k = torch.cat([k, k_src], dim=1)
        k = rearrange(k, 'fb three h w d -> fb (three h w) d', three=3) # b, s, d

        v = torch.cat([v, v_src], dim=1)
        v = rearrange(v, 'fb three h w d -> fb (three h w) d', three=3) # b, s, d

        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        # x: (b_half f) (three h w) d


        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context,         
        cam_intrinsic, # [batch, 4]
        cam_extrinsic, # [batch, compressed_frames, 12]        
        t_mod, freqs,        
        size_frame, size_height, size_width, # that of latents
        #no_warp=False,
        #no_src_intrinsic=False,        
        ):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)

        if hasattr(self, 'cam_encoder') :            
            
            # homography warping
            intrinsic_cond, intrinsic_trg = cam_intrinsic
            size_batch = intrinsic_trg.shape[0]
            freqs_3f, freqs_2b = freqs
            intrinsic_trg = intrinsic_trg/(intrinsic_trg[:, 2:3]*2) # normalize using image width
            intrinsic_cond = intrinsic_cond/(intrinsic_cond[:, 2:3]*2) # normalize using image width

            cam_emb_krt = self.cam_encoder(torch.cat([intrinsic_trg.unsqueeze(1).repeat(1, size_frame, 1), cam_extrinsic], dim=-1)) # -> [batch, compressed_frames, 1536]
            
            extrinsic_It = cam_extrinsic.clone()
            extrinsic_It[:, :, 0] = 1
            extrinsic_It[:, :, 1] = 0
            extrinsic_It[:, :, 2] = 0
            extrinsic_It[:, :, 4] = 0
            extrinsic_It[:, :, 5] = 1
            extrinsic_It[:, :, 6] = 0
            extrinsic_It[:, :, 8] = 0
            extrinsic_It[:, :, 9] = 0
            extrinsic_It[:, :, 10] = 1
            cam_emb_kIt = self.cam_encoder(torch.cat([intrinsic_trg.unsqueeze(1).repeat(1, size_frame, 1), extrinsic_It], dim=-1)) # -> [batch, compressed_frames, 1536]

            extrinsic_I0 = extrinsic_It.clone()
            extrinsic_I0[:, :, 3] = 0
            extrinsic_I0[:, :, 7] = 0
            extrinsic_I0[:, :, 11] = 0
            cam_emb_kI0 = self.cam_encoder(torch.cat([intrinsic_cond.unsqueeze(1).repeat(1, size_frame, 1), extrinsic_I0], dim=-1)) # -> [batch, compressed_frames, 1536]

            # input_x & frame to batch, batch to frame. 
            shift_msa0, scale_msa0, gate_msa0, shift_mlp0, scale_mlp0, gate_mlp0 = (
                self.modulation0.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)

            input_x = modulate(self.norm0(x), shift_msa0, scale_msa0)                       
            input_x = rearrange(input_x, '(two b) (f h w) d -> two b f h w d', two=2, f=size_frame, h=size_height, w=size_width)     # trg || src       
                         
            first_latent = input_x[1, :, 0:1, :, :, :].repeat(1, size_frame, 1, 1, 1)
            first_latent_warped =self.warp_latent_using_homography(first_latent, intrinsic_cond, intrinsic_trg, cam_extrinsic)                        
            
            first_latent_warped = rearrange(first_latent_warped, 'b f h w d -> b d f h w') 
            first_latent_warped = self.zero_conv(first_latent_warped)
            first_latent_warped = rearrange(first_latent_warped, 'b d f h w -> b f h w d')  

            first_latent = first_latent + first_latent_warped

            first_latent = first_latent + cam_emb_kIt.unsqueeze(0).unsqueeze(3).unsqueeze(4)            
            input_x[0:1, ...] = input_x[0:1, ...] + cam_emb_krt.unsqueeze(0).unsqueeze(3).unsqueeze(4) # trg
            input_x[1:2, ...] = input_x[1:2, ...] + cam_emb_kI0.unsqueeze(0).unsqueeze(3).unsqueeze(4) # cond(=src)
        
            input_x = torch.cat([input_x, first_latent], dim=0)  
                   
                
            input_x = rearrange(input_x, 'three b f h w d -> (b f) three h w d')  # three = trg, src, init frame         
            
            # triplet attention
            self_attn_pre_output = self.self_attn_pre(input_x, freqs_3f)

            # triplet -> pair, the remaining layers do not need the first image condition
            self_attn_pre_output = rearrange(self_attn_pre_output, 'bf (three h w) d -> bf three h w d', three=3, h=size_height, w=size_width)
            self_attn_pre_output = self_attn_pre_output[:, :2, ...]            
            self_attn_pre_output = rearrange(self_attn_pre_output, 'bf two h w d -> bf (two h w) d')

            x = rearrange(x, '(two b) (f h w) d -> two b f h w d', two=2, f=size_frame, h=size_height, w=size_width)            
            x = rearrange(x, 'two b f h w d -> (b f) (two h w) d')

            x = x + gate_msa0 * self_attn_pre_output    
            # x = x + gate_msa * self.projector(self.self_attn( input_x, freqs))
            
            # prepare to return to self attn input (batch -> frame, pair -> batch)
            x = rearrange(x, '(b f) (two h w) d -> (two b) (f h w) d', two=2, f=size_frame, h=size_height, w=size_width)            
            
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)        
            x = x + gate_msa * self.self_attn(input_x, freqs_2b)   

        else:    
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            x = x + gate_msa * self.self_attn(input_x, freqs)        
        
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x

    def safe_inverse(self, matrix):
        """BFloat16 안전한 역행렬 계산"""
        original_dtype = matrix.dtype
        try:
            if original_dtype == torch.bfloat16:
                inv_f32 = torch.linalg.inv(matrix.float())
                return inv_f32.to(original_dtype)
            else:
                return torch.linalg.inv(matrix)
        except torch.linalg.LinAlgError:
            print("Warning: Singular matrix detected, using pseudo-inverse")
            if original_dtype == torch.bfloat16:
                pinv_f32 = torch.linalg.pinv(matrix.float())
                return pinv_f32.to(original_dtype)
            else:
                return torch.linalg.pinv(matrix)

    def warp_latent_using_homography(self, latent_src, intrinsic_cond, intrinsic_trg, extrinsic):
        # latent_src: torch.Size([batch, compressed_frames, compressed_h, compressed_w, d])
        # intrinsic: torch.Size([batch, 4])
        # extrinsic: torch.Size([batch, compressed_frames, 12])
        
        b, f, h, w, d = latent_src.shape

        # unify the data type and device
        target_dtype = latent_src.dtype
        device = latent_src.device

        # create the intrinsic matrix (correct dtype)
        intrinsic_cond_mat = torch.zeros(b, 1, 3, 3, device=device, dtype=target_dtype)
        intrinsic_cond_mat[:, 0, 0, 0] = intrinsic_cond[:, 0].to(target_dtype) * w
        intrinsic_cond_mat[:, 0, 1, 1] = intrinsic_cond[:, 1].to(target_dtype) * w
        intrinsic_cond_mat[:, 0, 0, 2] = w/2 #intrinsic[:, 2].to(target_dtype) * w
        intrinsic_cond_mat[:, 0, 1, 2] = h/2 #intrinsic[:, 3].to(target_dtype) * w
        intrinsic_cond_mat[:, 0, 2, 2] = 1

        intrinsic_trg_mat = torch.zeros(b, 1, 3, 3, device=device, dtype=target_dtype)
        intrinsic_trg_mat[:, 0, 0, 0] = intrinsic_trg[:, 0].to(target_dtype) * w
        intrinsic_trg_mat[:, 0, 1, 1] = intrinsic_trg[:, 1].to(target_dtype) * w
        intrinsic_trg_mat[:, 0, 0, 2] = w/2 #intrinsic[:, 2].to(target_dtype) * w
        intrinsic_trg_mat[:, 0, 1, 2] = h/2 #intrinsic[:, 3].to(target_dtype) * w
        intrinsic_trg_mat[:, 0, 2, 2] = 1
        
        # create the rotation matrix (correct dtype)
        rots_mat = extrinsic.reshape(b, f, 3, 4)[:, :, :3, :3].transpose(-1, -2).to(target_dtype)
        
        # calculate the homography (safe inverse)
        intrinsic_cond_inv = self.safe_inverse(intrinsic_cond_mat)
        H = intrinsic_trg_mat @ rots_mat @ intrinsic_cond_inv
        
        # adjust the shape of the video tensor to input to grid_sample        
        latent_batch = latent_src.permute(0, 1, 4, 2, 3).reshape(b * f, d, h, w)

        # create the basic grid in the pixel coordinate system (0~w-1, 0~h-1)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=target_dtype),
            torch.arange(w, device=device, dtype=target_dtype),
            indexing='ij'
        )
        
        # create the homogeneous coordinate [h, w, 3]
        base_grid_h = torch.stack([
            x_coords,
            y_coords,
            torch.ones_like(x_coords)
        ], dim=-1)

        # reshape the shape to multiply the matrix (1, 3, h*w)
        grid_h_T = base_grid_h.view(h * w, 3).t().unsqueeze(0)

        # H inverse (safe inverse)
        H_inv = self.safe_inverse(H)

        # apply the inverse homography to calculate the source pixel coordinates
        # all operations are performed in the pixel coordinate system
        source_pixel_coords_h_T = H_inv @ grid_h_T
        source_pixel_coords_h = source_pixel_coords_h_T.transpose(-1, -2)

        # 8. perspective division (improved: adjust the clamp value)
        w_prime = source_pixel_coords_h[..., 2:3].clamp(min=1e-6)  
        source_pixel_coords_xy = source_pixel_coords_h[..., :2] / w_prime

        # normalize
        sampler_x = (source_pixel_coords_xy[..., 0] / (w - 1)) * 2 - 1
        sampler_y = (source_pixel_coords_xy[..., 1] / (h - 1)) * 2 - 1

        # create the final grid
        final_grid = torch.stack([sampler_x, sampler_y], dim=-1).view(b*f, h, w, 2)

        # Grid sample
        warped_latent_batch = F.grid_sample(
            latent_batch,
            final_grid, # -1 ~ 1 에 근접해야.
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        # restore the original shape
        warped_latent = warped_latent_batch.view(b, f, d, h, w).permute(0, 1, 3, 4, 2).to(latent_src.dtype)

        # check the final result
        if False:
            print("Warping complete with corrected coordinate system.")
            print("Warped video shape:", warped_latent.shape)
            print("Final grid min/max:", final_grid.min().item(), final_grid.max().item())

        return warped_latent


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    
    def forward(self,
            x: torch.Tensor,
            timestep: torch.Tensor,

            cam_intrinsic: torch.Tensor,
            cam_extrinsic: torch.Tensor,
            context: torch.Tensor,
            clip_feature: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            use_gradient_checkpointing: bool = False,
            use_gradient_checkpointing_offload: bool = False,
            **kwargs,
            ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs_3f = torch.cat([            
            self.freqs[0][:3].view(3, 1, 1, -1).expand(3, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(3, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(3, h, w, -1)
        ], dim=-1).reshape(3 * h * w, 1, -1).to(x.device)
        
        freqs_2b = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for i, block in enumerate(self.blocks):           
            freqs = (freqs_3f, freqs_2b)
            
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context,                             
                            cam_intrinsic, 
                            cam_extrinsic,                            
                            t_mod, freqs,                            
                            f, h, w, 
                            use_reentrant=False,                            
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context,                         
                        cam_intrinsic, 
                        cam_extrinsic,                        
                        t_mod, freqs,                        
                        f, h, w, 
                        use_reentrant=False,

                    )
            else:
                x = block(x, context,                             
                            cam_intrinsic, 
                            cam_extrinsic,                            
                            t_mod, freqs,                          
                            f, h, w, 
                          )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
