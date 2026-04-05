"""
BIT model wrapper with CoSA (Cross-correlation guided attention) integration.
Implements the "Freeze and Refine" strategy from recommended_way.md.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = RESEARCH_ROOT.parent


def _resolve_bit_dir():
    env_path = os.environ.get('BIT_CD_DIR', '').strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        RESEARCH_ROOT / 'checkpoints_official' / 'BIT_LEVIR' / 'BIT_CD-master',
        WORKSPACE_ROOT / 'checkpoints_official' / 'BIT_LEVIR' / 'BIT_CD-master',
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_cosa_block_class():
    """Load CoSABlockCD from any available local module copy."""
    candidates = [
        WORKSPACE_ROOT / 'models' / 'siamese_unet.py',
        RESEARCH_ROOT / 'models' / 'siamese_unet.py',
        RESEARCH_ROOT / 'configs' / 'custom' / 'cosa_v3' / 'siamese_unet.py',
        RESEARCH_ROOT / 'configs' / 'custom' / 'baseline' / 'siamese_unet.py',
    ]
    for module_path in candidates:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            f'cosa_siamese_unet_{module_path.stem}',
            str(module_path),
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'CoSABlockCD'):
            return module.CoSABlockCD
    raise ImportError(
        'Could not find CoSABlockCD. Expected one of: '
        + ', '.join(str(p) for p in candidates)
    )


BIT_DIR = _resolve_bit_dir()
CoSABlockCD = _load_cosa_block_class()

# We'll import BIT modules inside the function to avoid conflicts


class BITWithCoSA(nn.Module):
    """
    BIT model with CoSA block integration.
    
    Strategy:
    1. Freeze the original BIT model
    2. Extract features after transformer decoder (before final classifier)
    3. Apply CoSA to the difference features
    4. Combine with learnable gate (initialized to 0.0)
    """
    
    def __init__(self, bit_model, use_cosa=True, cosa_topk=32, gamma_init=0.0):
        """
        Args:
            bit_model: Pre-trained BIT model (BASE_Transformer)
            use_cosa: Whether to use CoSA (default: True)
            cosa_topk: Top-k for CoSA correlation (default: 32)
            gamma_init: Initial value for learnable gate (default: 0.0)
        """
        super().__init__()
        
        # Freeze the BIT backbone
        self.bit_model = bit_model
        for param in self.bit_model.parameters():
            param.requires_grad = False
        self.bit_model.eval()
        
        self.use_cosa = use_cosa
        
        if use_cosa:
            # CoSA block works on the difference features (32 channels, 1/8 resolution)
            # After transformer decoder, before upsampling
            cosa_channels = 32  # Output channels from forward_single
            self.cosa_block = CoSABlockCD(
                in_channels=cosa_channels,
                kernel_size=3,
                topk=cosa_topk,
                use_learnable_gate=True,
                gamma_init=gamma_init
            )
    
    def forward(self, x1, x2):
        """
        Forward pass with CoSA enhancement.
        
        Args:
            x1: [B, 3, H, W] - T1 image
            x2: [B, 3, H, W] - T2 image
        
        Returns:
            logits: [B, 2, H, W] - Change detection logits
        """
        # Get original BIT output (frozen)
        with torch.no_grad():
            original_logits = self.bit_model(x1, x2)  # [B, 2, H, W]
        
        if not self.use_cosa:
            return original_logits
        
        # Extract features for CoSA (need gradients for training CoSA)
        feat1 = self.bit_model.forward_single(x1)  # [B, 32, H/8, W/8]
        feat2 = self.bit_model.forward_single(x2)  # [B, 32, H/8, W/8]
        
        # Forward through tokenizer and transformer (no grad for frozen parts)
        with torch.no_grad():
            if self.bit_model.tokenizer:
                token1 = self.bit_model._forward_semantic_tokens(feat1)
                token2 = self.bit_model._forward_semantic_tokens(feat2)
            else:
                token1 = self.bit_model._forward_reshape_tokens(feat1)
                token2 = self.bit_model._forward_reshape_tokens(feat2)
            
            # Transformer encoder
            if self.bit_model.token_trans:
                tokens_ = torch.cat([token1, token2], dim=1)
                tokens = self.bit_model._forward_transformer(tokens_)
                token1, token2 = tokens.chunk(2, dim=1)
            
            # Transformer decoder
            if self.bit_model.with_decoder:
                feat1_decoded = self.bit_model._forward_transformer_decoder(feat1, token1)
                feat2_decoded = self.bit_model._forward_transformer_decoder(feat2, token2)
            else:
                feat1_decoded = self.bit_model._forward_simple_decoder(feat1, token1)
                feat2_decoded = self.bit_model._forward_simple_decoder(feat2, token2)
        
        # Feature differencing (this is where change information is)
        diff_feat = torch.abs(feat1_decoded - feat2_decoded)  # [B, 32, H/8, W/8]
        
        # Apply CoSA to enhance difference features
        diff_feat_enhanced, change_gate = self.cosa_block(
            feat1_decoded, 
            feat2_decoded, 
            diff_feat
        )
        
        # Upsample the enhanced features (need gradients for backprop through CoSA)
        if not self.bit_model.if_upsample_2x:
            diff_feat_enhanced = self.bit_model.upsamplex2(diff_feat_enhanced)
        diff_feat_enhanced = self.bit_model.upsamplex4(diff_feat_enhanced)
        
        # Use the frozen classifier (frozen but allows gradients to flow through)
        # We detach the classifier computation but keep gradients for CoSA
        enhanced_logits = self.bit_model.classifier(diff_feat_enhanced)
        
        # The CoSA block's gamma (initialized to 0.0) ensures we start at baseline
        # As training progresses, gamma learns to weight the CoSA enhancement
        return enhanced_logits
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (CoSA block)."""
        if self.use_cosa:
            return list(self.cosa_block.parameters())
        return []


def load_bit_with_cosa(checkpoint_path, device, use_cosa=True, cosa_topk=32, gamma_init=0.0):
    """
    Load BIT model and wrap it with CoSA.
    
    Args:
        checkpoint_path: Path to BIT checkpoint
        device: Torch device
        use_cosa: Whether to use CoSA
        cosa_topk: Top-k for CoSA
        gamma_init: Initial gate value
    
    Returns:
        BITWithCoSA model
    """
    if BIT_DIR is None:
        raise FileNotFoundError(
            'BIT_CD-master directory not found. '
            'Set BIT_CD_DIR or place it under checkpoints_official/BIT_LEVIR/BIT_CD-master.'
        )

    bit_dir_str = str(BIT_DIR)
    # Ensure BIT_DIR is FIRST in sys.path before importing BIT modules.
    if bit_dir_str in sys.path:
        sys.path.remove(bit_dir_str)
    sys.path.insert(0, bit_dir_str)
    
    # Remove cached models modules to force reload from BIT_DIR
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('models.')]
    for k in modules_to_remove:
        del sys.modules[k]
    if 'models' in sys.modules:
        del sys.modules['models']
    
    # Now import from BIT (will use BIT_DIR/models)
    from models.networks import define_G
    
    # Create args object for model definition
    class Args:
        def __init__(self):
            self.net_G = 'base_transformer_pos_s4_dd8_dedim8'
            self.gpu_ids = [0] if torch.cuda.is_available() else []
    
    args = Args()
    
    # Define and load BIT model
    bit_model = define_G(args=args, gpu_ids=args.gpu_ids)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle state dict
    if 'model_G_state_dict' in checkpoint:
        state_dict = checkpoint['model_G_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    bit_model.load_state_dict(state_dict, strict=True)
    bit_model.to(device)
    
    # Wrap with CoSA
    model = BITWithCoSA(
        bit_model=bit_model,
        use_cosa=use_cosa,
        cosa_topk=cosa_topk,
        gamma_init=gamma_init
    )
    model.to(device)
    
    return model
