# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Multi-Keyframe Video Generation Module

This module provides tools for generating long-form videos from multiple keyframes
using the Wan2.1 FLF2V (First-Last-Frame-to-Video) model.
"""

from .keyframe_manager import (
    KeyframeInfo,
    SegmentConfig,
    GenerationConfig,
    KeyframeManager,
)
from .segment_generator import (
    FLF2VSegmentGenerator,
    ParallelFLF2VGenerator,
)
from .video_stitcher import VideoStitcher
from .video_smoother import VideoSmoother
from .pipeline import MultiKeyframePipeline

__all__ = [
    # Data structures
    "KeyframeInfo",
    "SegmentConfig",
    "GenerationConfig",
    # Core classes
    "KeyframeManager",
    "FLF2VSegmentGenerator",
    "ParallelFLF2VGenerator",
    "VideoStitcher",
    "VideoSmoother",
    "MultiKeyframePipeline",
]

__version__ = "0.1.0"
