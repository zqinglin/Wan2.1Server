#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Multi-Keyframe Video Generation CLI Tool

Command-line interface for generating videos from multiple keyframes
using the Wan2.1 FLF2V model.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add parent directory to path to import wan module
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.multi_keyframe import (
    MultiKeyframePipeline,
    GenerationConfig,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-keyframe video generation using Wan2.1 FLF2V",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with keyframes
  python scripts/multi_keyframe_generate.py \\
    --keyframes "frame1.png:0.0,frame2.png:3.0,frame3.png:6.0" \\
    --prompt "cinematic style, soft lighting, 4k" \\
    --ckpt_dir ./Wan2.1-FLF2V-14B-720P

  # Using a config file
  python scripts/multi_keyframe_generate.py --config config.yaml

  # Parallel generation with multiple GPUs
  python scripts/multi_keyframe_generate.py \\
    --config config.yaml \\
    --num_gpus 4 \\
    --gpu_ids 0,1,2,3
        """
    )

    # Input configuration
    input_group = parser.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (alternative to CLI args)"
    )
    input_group.add_argument(
        "--keyframes",
        type=str,
        help="Keyframe list in format: 'path1:time1,path2:time2,...' "
             "(time in seconds)"
    )
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Global prompt for all segments"
    )
    input_group.add_argument(
        "--sub_prompts",
        type=str,
        help="Segment-specific prompts separated by '|'"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--ckpt_dir",
        type=str,
        help="Path to FLF2V model checkpoint directory"
    )
    model_group.add_argument(
        "--size",
        type=str,
        default="1280*720",
        help="Video size (width*height). Default: 1280*720"
    )
    model_group.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS. Default: 24"
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Number of sampling steps (default: auto)"
    )
    gen_group.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor (default: auto)"
    )
    gen_group.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale. Default: 5.0"
    )
    gen_group.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Base random seed (-1 for random). Default: -1"
    )

    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel generation. Default: 1"
    )
    perf_group.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Specific GPU IDs to use (comma-separated, e.g., '0,1,2,3')"
    )
    perf_group.add_argument(
        "--offload_model",
        action="store_true",
        help="Offload model to CPU to save VRAM"
    )
    perf_group.add_argument(
        "--t5_cpu",
        action="store_true",
        help="Keep T5 encoder on CPU"
    )

    # Post-processing options
    post_group = parser.add_argument_group("Post-Processing Options")
    post_group.add_argument(
        "--enable_smoothing",
        action="store_true",
        default=True,
        help="Enable post-processing smoothing (default: enabled)"
    )
    post_group.add_argument(
        "--no_smoothing",
        action="store_true",
        help="Disable post-processing smoothing"
    )
    post_group.add_argument(
        "--smoothing_method",
        type=str,
        default="temporal_filter",
        choices=["temporal_filter", "optical_flow", "none"],
        help="Smoothing method to use. Default: temporal_filter"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        type=str,
        default="./output_final.mp4",
        help="Output video path. Default: ./output_final.mp4"
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for segments and intermediate files. Default: ./outputs"
    )
    output_group.add_argument(
        "--keep_intermediates",
        action="store_true",
        help="Keep intermediate files (segments, stitched video)"
    )

    # Advanced options
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regeneration of existing segments"
    )
    adv_group.add_argument(
        "--resume_from",
        type=int,
        default=None,
        help="Resume generation from specific segment ID"
    )
    adv_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(args: argparse.Namespace) -> dict:
    """
    Merge command-line arguments with config file.

    Command-line arguments take precedence over config file.
    """
    config = {}

    # Load from YAML if provided
    if args.config:
        config = load_config_from_yaml(args.config)
        logging.info(f"Loaded configuration from: {args.config}")

    # Get parser defaults to check which args were explicitly provided
    parser = argparse.ArgumentParser()
    # We need to rebuild the parser to get defaults (simplified approach)
    # Only override if the value differs from default OR if no config was loaded

    # Override with command-line arguments (only if explicitly provided)
    # Special handling: only override if value is not the default or if key not in config
    defaults = {
        'size': '1280*720',
        'fps': 24,
        'sample_guide_scale': 5.0,
        'base_seed': -1,
        'num_gpus': 1,
        'smoothing_method': 'temporal_filter',
        'output': './output_final.mp4',
        'output_dir': './outputs',
    }

    for key, value in vars(args).items():
        if key == 'config':
            continue

        # Special handling for boolean flags
        if key == 'no_smoothing' and value:
            config['enable_smoothing'] = False
            continue

        # Only override if:
        # 1. Key is not in config (not from YAML), OR
        # 2. Value is different from default (explicitly provided by user)
        if key not in config or (value is not None and value != defaults.get(key)):
            if value is not None:
                config[key] = value

    return config


def parse_keyframes(keyframes_str: str) -> list:
    """
    Parse keyframes string into list of (path, timestamp) tuples.

    Args:
        keyframes_str: String in format "path1:time1,path2:time2,..."

    Returns:
        List of (path, timestamp) tuples
    """
    keyframes = []
    for item in keyframes_str.split(','):
        parts = item.strip().split(':')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid keyframe format: '{item}'. "
                f"Expected format: 'path:timestamp'"
            )
        path, time = parts
        keyframes.append((path.strip(), float(time.strip())))
    return keyframes


def validate_config(config: dict):
    """Validate configuration"""
    required_fields = ['keyframes', 'prompt', 'ckpt_dir']

    for field in required_fields:
        if field not in config or config[field] is None:
            raise ValueError(f"Required field missing: {field}")


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Merge configurations
        config = merge_configs(args)

        # Validate configuration
        validate_config(config)

        # Parse keyframes
        if isinstance(config['keyframes'], str):
            config['keyframes'] = parse_keyframes(config['keyframes'])

        # Parse sub-prompts
        sub_prompts = None
        if config.get('sub_prompts'):
            if isinstance(config['sub_prompts'], str):
                sub_prompts = [p.strip() for p in config['sub_prompts'].split('|')]
            else:
                sub_prompts = config['sub_prompts']

        # Parse GPU IDs
        gpu_ids = None
        if config.get('gpu_ids'):
            if isinstance(config['gpu_ids'], str):
                gpu_ids = [int(x.strip()) for x in config['gpu_ids'].split(',')]
            else:
                gpu_ids = config['gpu_ids']

        # Debug: print num_gpus from config
        logging.info(f"DEBUG: config['num_gpus'] = {config.get('num_gpus')}")

        # Create generation configuration
        gen_config = GenerationConfig(
            ckpt_dir=Path(config['ckpt_dir']),
            size=config.get('size', '1280*720'),
            fps=config.get('fps', 24),
            sample_steps=config.get('sample_steps'),
            sample_shift=config.get('sample_shift'),
            sample_guide_scale=config.get('sample_guide_scale', 5.0),
            base_seed=config.get('base_seed', -1),
            offload_model=config.get('offload_model', False),
            t5_cpu=config.get('t5_cpu', False),
            output_dir=Path(config.get('output_dir', './outputs'))
        )

        # Create and run pipeline
        pipeline = MultiKeyframePipeline(gen_config)

        output_path = pipeline.run(
            keyframes=config['keyframes'],
            global_prompt=config['prompt'],
            sub_prompts=sub_prompts,
            output_path=config.get('output', './output_final.mp4'),
            num_gpus=config.get('num_gpus', 1),
            gpu_ids=gpu_ids,
            enable_smoothing=config.get('enable_smoothing', True),
            smoothing_method=config.get('smoothing_method', 'temporal_filter'),
            force_regenerate=config.get('force_regenerate', False),
            resume_from=config.get('resume_from'),
            keep_intermediates=config.get('keep_intermediates', False)
        )

        print(f"\n✅ Success! Video saved to: {output_path}")
        return 0

    except Exception as e:
        logging.error(f"❌ Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
