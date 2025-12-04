import torch
from timm.layers import drop
from torch import nn
import math
from mamba_ssm.modules.mamba_simple import Mamba
from functools import partial
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
except ImportError:
    RMSNorm, LayerNorm = None, None
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,
            layer_scale=None,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"


    def forward(
            self, desc, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return desc + hidden_states



def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)



    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)



def efficient_scan(x, step_size=2, i=0): # [B, C, H, W] -> [B, 4, H/w * W/w]
    B, C, org_h, org_w = x.shape


    if org_w % step_size != 0:
        pad_w = step_size - org_w % step_size
        x = F.pad(x, (0, pad_w, 0, 0))
    W = x.shape[3]

    if org_h % step_size != 0:
        pad_h = step_size - org_h % step_size
        x = F.pad(x, (0, 0, 0, pad_h))
    H = x.shape[2]

    H = H // step_size
    W = W // step_size

    xs = x.new_empty((B, 4, C, H*W))

    # neurips veersion
    xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
    xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
    xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
    xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

    xs = xs.view(B, 4, -1, C)
    return xs, org_h, org_w


def efficient_merge(ys, ori_h, ori_w, step_size=2,i=0):
    B, K, L, C = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y = ys.new_empty((B, C, new_h, new_w))

    # neurips veersion
    y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
    y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
    y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
    y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y = y[:, :, :ori_h, :ori_w].contiguous()

    # y = y.view(B, C, -1)
    return y



def reference_centered_dynamic_scanning(feat0, feat1, step_size, i=0):
    feats_right_2w = torch.cat([feat0, feat1], dim=3)  # feat1 在右侧
    feats_left_2w = torch.cat([feat1, feat0], dim=3)  # feat1 在左侧
    feats_below_2h = torch.cat([feat0, feat1], dim=2)  # feat1 在下方
    feat_top_2h = torch.cat([feat1, feat0], dim=2)  # feat1 在上方

    _, _, org_h, org_2w = feats_right_2w.shape
    B, C, org_2h, org_w = feats_below_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    feats = feats_right_2w.new_empty((B, 4, C, H * W))

    if i == 0:
        feats[:, 0] = feats_right_2w[:, :, 1::step_size, ::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)
        feats[:, 1] = feats_left_2w[:, :, ::step_size, ::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C,-1).flip([2])
        feats[:, 2] = feats_below_2h[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        feats[:, 3] = feat_top_2h[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])
    elif i == 1:
        feats[:, 0] = feats_right_2w[:, :, ::step_size, ::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)
        feats[:, 1] = feats_left_2w[:, :, 1::step_size, ::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C,-1).flip([2])
        feats[:, 2] = feats_below_2h[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)
        feats[:, 3] = feat_top_2h[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])
    elif i == 2:
        feats[:, 0] = feats_right_2w[:, :, ::step_size, 1::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)
        feats[:, 1] = feats_left_2w[:, :, 1::step_size, 1::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C,-1).flip([2])
        feats[:, 2] = feats_below_2h[:, :, 1::step_size, ::step_size].contiguous().view(B, C, -1)
        feats[:, 3] = feat_top_2h[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1).flip([2])
    else:
        feats[:, 0] = feats_right_2w[:, :, 1::step_size, 1::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)
        feats[:, 1] = feats_left_2w[:, :, ::step_size, 1::step_size].transpose(dim0=2, dim1=3).contiguous().view(B, C,-1).flip([2])
        feats[:, 2] = feats_below_2h[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        feats[:, 3] = feat_top_2h[:, :, 1::step_size, ::step_size].contiguous().view(B, C, -1).flip([2])

    feats = feats.view(B, 4, C, -1).transpose(2, 3)
    return feats, org_h, org_w


def feature_merging(ys, ori_h, ori_w, step_size, i=0):
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w_right = torch.zeros((B, C, new_h, 2 * new_w), device=ys.device, dtype=ys.dtype)
    y_2w_left = torch.zeros((B, C, new_h, 2 * new_w), device=ys.device, dtype=ys.dtype)
    y_2h_below = torch.zeros((B, C, 2 * new_h, new_w), device=ys.device, dtype=ys.dtype)
    y_2h_top = torch.zeros((B, C, 2 * new_h, new_w), device=ys.device, dtype=ys.dtype)

    if i == 0:
        y_2w_right[:, :, 1::step_size, ::step_size] = ys[:, 0].reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2w_left[:, :, ::step_size, ::step_size] = ys[:, 1].flip([2]).reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2h_below[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, 2 * H, W)
        y_2h_top[:, :, 1::step_size, 1::step_size] = ys[:, 3].flip([2]).reshape(B, C, 2 * H, W)
    elif i == 1:
        y_2w_right[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2w_left[:, :, 1::step_size, ::step_size] = ys[:, 1].flip([2]).reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2h_below[:, :, 1::step_size, 1::step_size] = ys[:, 2].reshape(B, C, 2 * H, W)
        y_2h_top[:, :, ::step_size, 1::step_size] = ys[:, 3].flip([2]).reshape(B, C, 2 * H, W)
    elif i == 2:
        y_2w_right[:, :, ::step_size, 1::step_size] = ys[:, 0].reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2w_left[:, :, 1::step_size, 1::step_size] = ys[:, 1].flip([2]).reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2h_below[:, :, 1::step_size, ::step_size] = ys[:, 2].reshape(B, C, 2 * H, W)
        y_2h_top[:, :, ::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, 2 * H, W)
    else:
        y_2w_right[:, :, 1::step_size, 1::step_size] = ys[:, 0].reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2w_left[:, :, ::step_size, 1::step_size] = ys[:, 1].flip([2]).reshape(B, C, 2 * W, H).transpose(dim0=2, dim1=3)
        y_2h_below[:, :, ::step_size, ::step_size] = ys[:, 2].reshape(B, C, 2 * H, W)
        y_2h_top[:, :, 1::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, 2 * H, W)

    if ori_h != new_h or ori_w != new_w:
        y_2w_right = y_2w_right[:, :, :ori_h, :ori_w].contiguous()
        y_2w_left = y_2w_left[:, :, :ori_h, :ori_w].contiguous()
        y_2h_top = y_2h_top[:, :, :ori_h, :ori_w].contiguous()
        y_2h_below = y_2h_below[:, :, :ori_h, :ori_w].contiguous()

    desc0_2w_right, desc1_2w_right = torch.chunk(y_2w_right, 2, dim=3)
    desc1_2w_left, desc0_2w_left = torch.chunk(y_2w_left, 2, dim=3)
    desc0_2h_below, desc1_2h_below = torch.chunk(y_2h_below, 2, dim=2)
    desc1_2h_top, desc0_2h_top = torch.chunk(y_2h_top, 2, dim=2)

    return desc0_2w_right + desc0_2w_left + desc0_2h_below + desc0_2h_top, \
           desc1_2w_right + desc1_2w_left + desc1_2h_below + desc1_2h_top


class DM(nn.Module):
    def __init__(self, feature_dim, depth,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 if_bimamba=False,
                 ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_layers = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(create_block(
                    feature_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                ))
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim, bias=False),
        )
        self.norm = nn.LayerNorm(feature_dim)


    def forward(self, ref_feat, src_feat, step_size=2, src_number=0):


        y, ori_h, ori_w = reference_centered_dynamic_scanning(ref_feat, src_feat,step_size,i=src_number%4)  # (B, 4, ori_h//2 * ori_w, C)


        for i in range(len(self.layers) // 4):
            y0 = self.layers[i * 4](y[:, 0])
            y1 = self.layers[i * 4 + 1](y[:, 1])
            y2 = self.layers[i * 4 + 2](y[:, 2])
            y3 = self.layers[i * 4 + 3](y[:, 3])
            y = torch.stack([y0, y1, y2, y3], 1)  # (B, 4, C, ori_h//2 * ori_w)
        y = self.norm(self.mlp(y)) + y

        ref_feat, src_feat = feature_merging(y.transpose(2, 3), ori_h, ori_w,step_size,i=src_number%4)

        return ref_feat, src_feat


class SDM(nn.Module):
    def __init__(self, feature_dim, depth,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 if_bimamba=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_layers = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(create_block(
                feature_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
            ))
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim, bias=False),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, feature, step_size=2, src_number=0):

        xs, H, W = efficient_scan(feature, step_size=step_size,i=src_number)
        y = xs
        for i in range(len(self.layers) // 4):
            y0 = self.layers[i * 4](y[:, 0])
            y1 = self.layers[i * 4 + 1](y[:, 1])
            y2 = self.layers[i * 4 + 2](y[:, 2])
            y3 = self.layers[i * 4 + 3](y[:, 3])
            y = torch.stack([y0, y1, y2, y3], dim=1)

        y = self.norm(self.mlp(y)) + y

        feature = efficient_merge(y, H, W, step_size=step_size)

        return feature




