import torch
import math
import copy
import logging
from typing import Tuple, Dict, Any, List, Optional

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.latent_formats
import latent_preview
import node_helpers

# Set up logging for FreeLong
logger = logging.getLogger("comfyui.longlook")
logger.setLevel(logging.INFO)


class WanContinuationConditioning:
    """
    Simple video continuation conditioning for Wan 2.2.

    Takes the last frame from a previous video chunk and creates i2v conditioning
    for generating the next chunk. Equivalent to WanImageToVideo but designed
    for easy chaining in continuation workflows.

    Usage: Connect decoded images from chunk 1 → this node → KSampler for chunk 2
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "anchor_images": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "video_length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1024,
                    "step": 4,
                    "tooltip": "Output video length in frames. Must match your sampler settings."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "modify_conditioning"
    CATEGORY = "video/wan"
    DESCRIPTION = "Creates i2v conditioning from last frame for video continuation"

    def modify_conditioning(
        self,
        positive,
        negative,
        anchor_images: torch.Tensor,
        vae,
        width: int,
        height: int,
        video_length: int
    ) -> Tuple[list, list, Dict[str, torch.Tensor]]:
        """
        Modify conditioning to use anchor frames for continuation.
        """
        import comfy.utils

        # Calculate latent dimensions using Wan's formula
        latent_frames = ((video_length - 1) // 4) + 1
        latent_h = height // 8
        latent_w = width // 8

        # Get the LAST frame from anchor images (single frame continuation like standard i2v)
        frames_to_encode = anchor_images[-1:]  # Always use just the last frame

        # Resize to target dimensions if needed
        if frames_to_encode.shape[1] != height or frames_to_encode.shape[2] != width:
            frames_to_encode = comfy.utils.common_upscale(
                frames_to_encode.movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)

        # Create full video tensor for VAE encoding (matches WanImageToVideo exactly)
        # Gray (0.5) for frames to generate, actual image for first frame
        full_video = torch.ones(
            video_length, height, width, 3,
            device=frames_to_encode.device, dtype=frames_to_encode.dtype
        ) * 0.5

        # Place the single anchor frame at the beginning
        full_video[:1] = frames_to_encode[:, :, :, :3]

        # VAE encode the entire video (anchor frames + gray neutral frames)
        concat_latent_image = vae.encode(full_video)
        if concat_latent_image.ndim == 4:
            concat_latent_image = concat_latent_image.unsqueeze(0)

        encoded_latent_frames = concat_latent_image.shape[2]
        latent_channels = concat_latent_image.shape[1]

        # Calculate how many latent frames to preserve (1 video frame = 1 latent frame)
        # Formula: ((1 - 1) // 4) + 1 = 1
        anchor_latent_frames = 1

        # Create concat_mask using LATENT frames (like WanImageToVideo does - simpler, no reshape)
        # Shape: [1, 1, latent_frames, h, w]
        # Wan convention: 0.0 = preserve (don't denoise), 1.0 = generate (full denoise)
        concat_mask = torch.ones(
            1, 1, encoded_latent_frames,
            concat_latent_image.shape[-2],
            concat_latent_image.shape[-1],
            device=frames_to_encode.device, dtype=frames_to_encode.dtype
        )

        # Preserve anchor latent frames
        concat_mask[:, :, :anchor_latent_frames] = 0.0

        # Move to CPU for conditioning storage
        concat_latent_image = concat_latent_image.cpu()
        concat_mask = concat_mask.cpu()

        # Modify conditioning with our anchor data
        # This overwrites any existing concat_latent_image/concat_mask from WanImageToVideo
        positive_out = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask
        })
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask
        })

        # Create empty latent for sampling
        latent = torch.zeros(
            1, latent_channels, latent_frames, latent_h, latent_w,
            device=comfy.model_management.intermediate_device()
        )

        return (positive_out, negative_out, {"samples": latent})


# ============================================================================
# FreeLong Helper Functions
# ============================================================================

def freelong_freq_filter(x: torch.Tensor, threshold: float, filter_type: str = "low") -> torch.Tensor:
    """
    Apply frequency filtering along the temporal (frame) dimension.

    Uses complementary filters that sum to 1.0 at all frequencies.

    Args:
        x: Tensor of shape [batch, frames, spatial, dim] or [batch, seq, dim]
        threshold: Cutoff frequency ratio (0.0 to 1.0)
        filter_type: "low" for low-pass, "high" for high-pass

    Returns:
        Filtered tensor
    """
    # cuFFT doesn't support half precision for non-power-of-two sizes
    # Convert to float32 for FFT, then back to original dtype
    orig_dtype = x.dtype
    x_float = x.float()

    # FFT along frame/sequence dimension
    freq = torch.fft.rfft(x_float, dim=1)

    # Create frequency mask (in float32)
    # Use smooth cosine transition that sums to 1.0 with complement
    num_freq = freq.shape[1]
    cutoff = int(num_freq * threshold)

    # Width of transition zone (at least 1, at most 25% of spectrum)
    transition_width = max(1, min(num_freq // 4, cutoff // 2))

    # Create smooth low-pass mask using cosine transition
    # This ensures low + high = 1.0 at all frequencies
    mask = torch.ones(num_freq, device=x.device, dtype=torch.float32)

    # Transition zone centered around cutoff
    transition_start = max(0, cutoff - transition_width // 2)
    transition_end = min(num_freq, cutoff + transition_width // 2)

    if transition_end > transition_start:
        # Cosine transition from 1 to 0 for low-pass
        transition_idx = torch.arange(transition_start, transition_end, device=x.device, dtype=torch.float32)
        transition_values = 0.5 * (1 + torch.cos(torch.pi * (transition_idx - transition_start) / (transition_end - transition_start)))
        mask[transition_start:transition_end] = transition_values

    # Set values outside transition
    mask[transition_end:] = 0.0

    if filter_type == "high":
        # High-pass is complement of low-pass: high = 1 - low
        mask = 1.0 - mask

    # Apply mask (keep in float32)
    mask = mask.view(1, -1, *([1] * (len(freq.shape) - 2)))
    filtered_freq = freq * mask

    # Inverse FFT and convert back to original dtype
    result = torch.fft.irfft(filtered_freq, n=x.shape[1], dim=1)
    return result.to(orig_dtype)


def freelong_spectral_blend(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    low_freq_ratio: float = 0.25,
    blend_strength: float = 1.0
) -> torch.Tensor:
    """
    Blend global and local features using spectral decomposition.
    Low frequencies from global (structure/motion direction),
    High frequencies from local (details/sharpness).

    Args:
        global_features: Features from full-sequence attention
        local_features: Features from windowed attention
        low_freq_ratio: Ratio of frequencies to take from global (default 0.25)
        blend_strength: How much to apply the blend (0=passthrough, 1=full blend)

    Returns:
        Blended features
    """
    if blend_strength == 0.0:
        return local_features

    # Get low-freq from global, high-freq from local
    global_low = freelong_freq_filter(global_features, low_freq_ratio, "low")
    local_high = freelong_freq_filter(local_features, low_freq_ratio, "high")

    # Blend with strength control
    blended = global_low + local_high

    # Mix with original based on blend_strength
    if blend_strength < 1.0:
        return local_features * (1.0 - blend_strength) + blended * blend_strength

    return blended


# ============================================================================
# FreeLong Model Patcher
# ============================================================================

class WanFreeLong:
    """
    FreeLong model patcher for Wan 2.2 video generation.

    Implements SpectralBlend temporal attention to extend video generation
    beyond the training context window while maintaining motion consistency.

    Based on "FreeLong: Training-Free Long Video Generation with
    SpectralBlend Temporal Attention" (NeurIPS 2024)

    How it works:
    1. Hooks into Wan's transformer blocks
    2. Runs attention twice: global (full sequence) and local (windowed)
    3. Blends results using FFT: low-freq from global, high-freq from local
    4. This preserves motion direction (low-freq) while keeping details (high-freq)
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Turn FreeLong on or off. Useful for comparing results with and without."
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How strongly FreeLong affects the video. Higher = more consistent motion but slightly softer. Lower = sharper but may drift. 0.7 is optimal for most cases."
                }),
                "low_freq_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Balance between motion consistency and fine detail. Higher = smoother, more consistent motion. Lower = more dynamic but may drift. 0.5 is a good balance."
                }),
                "local_window_frames": ("INT", {
                    "default": 33,
                    "min": 9,
                    "max": 241,
                    "step": 4,
                    "tooltip": "How many video frames the detail-preservation looks at. Smaller = sharper details but may cause morphing. Larger = smoother transitions. 33 works well for 81-frame videos. For longer videos, try ~40% of total frames."
                }),
                "blend_start_block": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 40,
                    "tooltip": "Advanced: Which model layer to start applying FreeLong. 0 = from the beginning. Most users should leave this at 0."
                }),
                "blend_end_block": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 40,
                    "tooltip": "Advanced: Which model layer to stop applying FreeLong. -1 = apply to all layers. Most users should leave this at -1."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "video/wan"
    DESCRIPTION = "Apply FreeLong spectral blending for better motion consistency in long videos"

    def patch_model(
        self,
        model,
        enabled: bool,
        blend_strength: float,
        low_freq_ratio: float,
        local_window_frames: int,
        blend_start_block: int,
        blend_end_block: int
    ):
        """
        Patch the model to use FreeLong spectral blending in temporal attention.
        """
        # Convert video frames to latent frames
        local_window_size = ((local_window_frames - 1) // 4) + 1
        # If disabled, return model unchanged (no patches)
        if not enabled:
            logger.info("[FreeLong] DISABLED - Model returned without patching")
            return (model,)

        logger.info("=" * 60)
        logger.info("[FreeLong] APPLYING SPECTRAL BLENDING PATCHES")
        logger.info("=" * 60)
        logger.info(f"  Blend Strength: {blend_strength:.2f}")
        logger.info(f"  Low Freq Ratio: {low_freq_ratio:.2f}")
        logger.info(f"  Local Window: {local_window_frames} video frames → {local_window_size} latent frames")
        logger.info(f"  Block Range: {blend_start_block} to {blend_end_block if blend_end_block != -1 else 'ALL'}")

        model = model.clone()

        # Store settings for the patch function
        freelong_settings = {
            "blend_strength": blend_strength,
            "low_freq_ratio": low_freq_ratio,
            "local_window_size": local_window_size,
            "blend_start_block": blend_start_block,
            "blend_end_block": blend_end_block,
            "first_run": True,  # Track first execution to log details once
            "blocks_patched": set(),  # Track which blocks executed
        }

        def freelong_block_patch(block_args, extra_args):
            """
            Patch function that wraps transformer blocks with FreeLong blending.

            Implements dual-stream processing:
            1. Global stream: Full sequence attention (original behavior)
            2. Local stream: Windowed attention (process temporal chunks)
            3. Spectral blend: Low-freq from global, high-freq from local
            """
            original_block = extra_args["original_block"]
            transformer_options = block_args.get("transformer_options", {})
            block_index = transformer_options.get("block_index", 0)
            total_blocks = transformer_options.get("total_blocks", 40)

            # Determine if this block should use FreeLong
            start = freelong_settings["blend_start_block"]
            end = freelong_settings["blend_end_block"]
            if end == -1:
                end = total_blocks

            # For blocks outside our range, just run original
            if block_index < start or block_index >= end:
                return original_block(block_args)

            # Skip if blend_strength is 0
            strength = freelong_settings["blend_strength"]
            if strength == 0.0:
                return original_block(block_args)

            # Track that this block executed (for logging summary)
            freelong_settings["blocks_patched"].add(block_index)

            try:
                x = block_args["img"]  # [batch, seq_len, dim]
                batch_size, seq_len, hidden_dim = x.shape

                window_size = freelong_settings["local_window_size"]
                low_ratio = freelong_settings["low_freq_ratio"]

                # Log details on first execution
                if freelong_settings["first_run"]:
                    freelong_settings["first_run"] = False
                    logger.info("")
                    logger.info("[FreeLong] DUAL-STREAM PROCESSING STARTING")
                    logger.info(f"  Input shape: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    logger.info(f"  Window size: {window_size} latent frames")

                # ============ GLOBAL STREAM ============
                # Full sequence attention - preserves global motion patterns
                global_out = original_block(block_args)
                global_x = global_out["img"]

                # ============ LOCAL STREAM ============
                # Process sequence in temporal windows for local detail preservation
                #
                # Wan's sequence is flattened [frames * spatial, dim]
                # We need to estimate the temporal structure
                # For 81 frame video: 21 latent frames, so spatial = seq_len / 21

                # Try to infer frame count from sequence length
                # Common Wan resolutions and their spatial token counts:
                # 1280x720 -> 160x90/4 = 40x22.5 -> ~900 tokens per frame (after 2x2 patch)
                # 848x480 -> 106x60/4 = 26.5x15 -> ~400 tokens per frame
                # We'll try common latent frame counts: 21, 17, 13, 9, 5

                num_frames = None
                for candidate_frames in [21, 17, 13, 9, 5]:
                    if seq_len % candidate_frames == 0:
                        num_frames = candidate_frames
                        break

                if num_frames is None or num_frames <= window_size:
                    # Can't determine temporal structure or window covers all frames
                    # Fall back to just using global output with spectral smoothing
                    if freelong_settings.get("log_fallback", True):
                        freelong_settings["log_fallback"] = False
                        if num_frames is None:
                            logger.warning(f"[FreeLong] Could not determine temporal structure (seq_len={seq_len})")
                        else:
                            logger.warning(f"[FreeLong] Window size >= frames ({window_size} >= {num_frames})")
                        logger.warning("[FreeLong] Using fallback: spectral smoothing without windowing")
                    blended_x = freelong_spectral_blend(global_x, global_x, low_ratio, strength)
                    return {"img": blended_x}

                spatial_size = seq_len // num_frames

                # Log temporal structure detection on first run
                if freelong_settings.get("log_structure", True):
                    freelong_settings["log_structure"] = False
                    logger.info(f"  Detected temporal structure: {num_frames} frames × {spatial_size} spatial tokens")

                # Reshape to [batch, frames, spatial, dim]
                x_temporal = x.view(batch_size, num_frames, spatial_size, hidden_dim)

                # Process in OVERLAPPING windows with crossfade blending
                # This prevents hard discontinuities at window boundaries

                # Get RoPE embeddings - need to slice these for each window too
                freqs = block_args.get("pe")  # RoPE positional embeddings

                # freqs shape is typically [batch, rope_dim, seq_len] or [1, seq_len, rope_dim]
                # We need to figure out which dim is the sequence dim
                freqs_seq_dim = None
                if freqs is not None:
                    for dim_idx, dim_size in enumerate(freqs.shape):
                        if dim_size == seq_len:
                            freqs_seq_dim = dim_idx
                            break

                # Use 50% overlap between windows for smooth blending
                overlap = window_size // 2
                stride = window_size - overlap

                # Calculate number of windows
                num_windows = 0
                temp_start = 0
                while temp_start < num_frames:
                    num_windows += 1
                    temp_start += stride
                    if temp_start >= num_frames:
                        break

                # Log windowing setup on first run
                if freelong_settings.get("log_windows", True):
                    freelong_settings["log_windows"] = False
                    logger.info(f"  Window processing: {num_windows} windows with {overlap} frame overlap")
                    logger.info(f"  Overlap strategy: 50% crossfade blending between windows")

                # Accumulator for blended output and weights
                local_x_temporal = torch.zeros(batch_size, num_frames, spatial_size, hidden_dim,
                                               device=x.device, dtype=x.dtype)
                weight_accumulator = torch.zeros(batch_size, num_frames, 1, 1,
                                                  device=x.device, dtype=x.dtype)

                # Process overlapping windows
                window_idx = 0
                start_frame = 0
                while start_frame < num_frames:
                    end_frame = min(start_frame + window_size, num_frames)
                    window_frames = end_frame - start_frame

                    # Extract window
                    window_x = x_temporal[:, start_frame:end_frame, :, :]

                    # Flatten for block processing
                    window_flat = window_x.reshape(batch_size, window_frames * spatial_size, hidden_dim)

                    # Create modified block_args for this window
                    window_args = dict(block_args)
                    window_args["img"] = window_flat

                    # Slice RoPE embeddings to match window
                    if freqs is not None and freqs_seq_dim is not None:
                        start_token = start_frame * spatial_size
                        end_token = end_frame * spatial_size
                        slices = [slice(None)] * len(freqs.shape)
                        slices[freqs_seq_dim] = slice(start_token, end_token)
                        window_args["pe"] = freqs[tuple(slices)]

                    # Run block on window
                    window_out = original_block(window_args)
                    window_result = window_out["img"]

                    # Reshape back to temporal
                    window_result = window_result.view(batch_size, window_frames, spatial_size, hidden_dim)

                    # Create blending weights - ramp up at start, ramp down at end
                    # This creates smooth crossfades in overlapping regions
                    weights = torch.ones(window_frames, device=x.device, dtype=x.dtype)

                    if start_frame > 0 and overlap > 0:
                        # Ramp up at the start (we're overlapping with previous window)
                        ramp_len = min(overlap, window_frames)
                        ramp = torch.linspace(0, 1, ramp_len + 1, device=x.device, dtype=x.dtype)[1:]
                        weights[:ramp_len] = ramp

                    if end_frame < num_frames and overlap > 0:
                        # Ramp down at the end (next window will overlap with us)
                        ramp_len = min(overlap, window_frames)
                        ramp = torch.linspace(1, 0, ramp_len + 1, device=x.device, dtype=x.dtype)[:-1]
                        weights[-ramp_len:] = weights[-ramp_len:] * ramp

                    weights = weights.view(1, window_frames, 1, 1)

                    # Accumulate weighted results
                    local_x_temporal[:, start_frame:end_frame] += window_result * weights
                    weight_accumulator[:, start_frame:end_frame] += weights

                    # Move to next window position
                    start_frame += stride
                    if start_frame >= num_frames:
                        break
                    window_idx += 1

                # Normalize by accumulated weights
                weight_accumulator = weight_accumulator.clamp(min=1e-8)
                local_x_temporal = local_x_temporal / weight_accumulator
                local_x = local_x_temporal.reshape(batch_size, seq_len, hidden_dim)

                # ============ SPECTRAL BLEND ============
                # Low frequencies from global (motion direction/consistency)
                # High frequencies from local (sharp details)
                if freelong_settings.get("log_blend", True):
                    freelong_settings["log_blend"] = False
                    logger.info("")
                    logger.info("[FreeLong] SPECTRAL BLENDING")
                    logger.info(f"  Low-freq (global): {low_ratio:.0%} of spectrum (motion consistency)")
                    logger.info(f"  High-freq (local): {1-low_ratio:.0%} of spectrum (sharp details)")
                    logger.info(f"  Blend strength: {strength:.0%}")

                blended_x = freelong_spectral_blend(global_x, local_x, low_ratio, strength)

                return {"img": blended_x}

            except Exception as e:
                # If anything fails, fall back to original block
                logger.error(f"[FreeLong] Error in block {block_index}: {e}", exc_info=True)
                return original_block(block_args)

        # Wan 2.2 typically has 40 blocks
        num_blocks = 40

        # Register the patch for transformer blocks
        for i in range(num_blocks):
            model.set_model_patch_replace(
                freelong_block_patch,
                "dit",
                "double_block",
                i
            )

        actual_end = blend_end_block if blend_end_block != -1 else num_blocks
        blocks_to_patch = actual_end - blend_start_block
        logger.info(f"  Registered patches for {blocks_to_patch} transformer blocks ({blend_start_block} to {actual_end-1})")
        logger.info("=" * 60)
        logger.info("[FreeLong] Patching complete. FreeLong will activate during sampling.")
        logger.info("=" * 60)

        return (model,)
