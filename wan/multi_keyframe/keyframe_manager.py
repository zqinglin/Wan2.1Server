# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Keyframe Management Module

Handles keyframe validation, segment configuration, and timeline management
for multi-keyframe video generation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class KeyframeInfo:
    """Information about a single keyframe"""
    frame_path: Path
    timestamp_sec: float
    frame_number: Optional[int] = None

    def __post_init__(self):
        """Ensure frame_path is a Path object"""
        if isinstance(self.frame_path, str):
            self.frame_path = Path(self.frame_path)


@dataclass
class SegmentConfig:
    """Configuration for a single video segment generation"""
    segment_id: int
    first_frame: Path
    last_frame: Path
    num_frames: int
    prompt: str
    sub_prompt: Optional[str] = None

    def __post_init__(self):
        """Ensure paths are Path objects"""
        if isinstance(self.first_frame, str):
            self.first_frame = Path(self.first_frame)
        if isinstance(self.last_frame, str):
            self.last_frame = Path(self.last_frame)

    @property
    def full_prompt(self) -> str:
        """Get the complete prompt (global + sub if exists)"""
        if self.sub_prompt:
            return f"{self.prompt}. {self.sub_prompt}"
        return self.prompt


@dataclass
class GenerationConfig:
    """Global configuration for video generation"""
    ckpt_dir: Path
    size: str = "1280*720"
    fps: int = 24
    sample_steps: Optional[int] = None
    sample_shift: Optional[float] = None
    sample_guide_scale: float = 5.0
    base_seed: int = -1
    offload_model: bool = False
    t5_cpu: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))

    def __post_init__(self):
        """Ensure paths are Path objects and directories exist"""
        if isinstance(self.ckpt_dir, str):
            self.ckpt_dir = Path(self.ckpt_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration"""
        # Check checkpoint directory
        if not self.ckpt_dir.exists():
            return False, f"Checkpoint directory not found: {self.ckpt_dir}"

        # Check required model files
        required_files = [
            "config.json",
            "Wan2.1_VAE.pth",
        ]
        for filename in required_files:
            if not (self.ckpt_dir / filename).exists():
                return False, f"Required file not found: {filename}"

        # Validate size format
        try:
            w, h = self.size.split('*')
            width, height = int(w), int(h)
            if width <= 0 or height <= 0:
                return False, f"Invalid size dimensions: {self.size}"
        except (ValueError, AttributeError):
            return False, f"Invalid size format: {self.size}. Expected format: 'width*height'"

        # Validate FPS
        if self.fps <= 0:
            return False, f"FPS must be positive: {self.fps}"

        return True, "Configuration is valid"


class KeyframeManager:
    """
    Manages keyframes and generates segment configurations for video generation.

    This class handles:
    - Validation of keyframe sequences
    - Frame number calculation based on FPS
    - Segment configuration generation
    - Duration and frame count recommendations
    """

    def __init__(
        self,
        keyframes: List[Tuple[str, float]],
        fps: int = 24
    ):
        """
        Initialize KeyframeManager.

        Args:
            keyframes: List of (frame_path, timestamp_sec) tuples
            fps: Target video frame rate
        """
        self.fps = fps
        self.keyframes = [
            KeyframeInfo(
                frame_path=Path(path),
                timestamp_sec=ts,
                frame_number=int(ts * fps)
            )
            for path, ts in keyframes
        ]

        logger.info(f"Initialized KeyframeManager with {len(self.keyframes)} keyframes at {fps} FPS")

    def validate(self) -> Tuple[bool, str]:
        """
        Validate the keyframe sequence.

        Returns:
            Tuple of (is_valid, message)
        """
        if len(self.keyframes) < 2:
            return False, "At least 2 keyframes are required"

        # Check file existence
        for i, kf in enumerate(self.keyframes):
            if not kf.frame_path.exists():
                return False, f"Keyframe {i} not found: {kf.frame_path}"

        # Check chronological order
        timestamps = [kf.timestamp_sec for kf in self.keyframes]
        if timestamps != sorted(timestamps):
            return False, "Keyframes must be in chronological order"

        # Check for duplicate timestamps
        if len(set(timestamps)) != len(timestamps):
            return False, "Duplicate timestamps found"

        # Check image dimensions consistency
        try:
            sizes = []
            for kf in self.keyframes:
                with Image.open(kf.frame_path) as img:
                    sizes.append(img.size)

            if len(set(sizes)) > 1:
                unique_sizes = set(sizes)
                return False, f"Inconsistent frame sizes: {unique_sizes}. All keyframes must have the same dimensions."
        except Exception as e:
            return False, f"Error reading keyframe images: {str(e)}"

        logger.info("✓ Keyframe validation passed")
        return True, "Validation passed"

    def get_keyframe_size(self) -> Tuple[int, int]:
        """Get the size (width, height) of keyframes"""
        if not self.keyframes:
            raise ValueError("No keyframes available")

        with Image.open(self.keyframes[0].frame_path) as img:
            return img.size

    def get_segments(
        self,
        global_prompt: str,
        sub_prompts: Optional[List[str]] = None
    ) -> List[SegmentConfig]:
        """
        Generate segment configurations from keyframes.

        Args:
            global_prompt: The main prompt describing the overall video
            sub_prompts: Optional list of segment-specific prompts

        Returns:
            List of SegmentConfig objects
        """
        if len(self.keyframes) < 2:
            raise ValueError("At least 2 keyframes are required to generate segments")

        segments = []

        for i in range(len(self.keyframes) - 1):
            kf_curr = self.keyframes[i]
            kf_next = self.keyframes[i + 1]

            # Calculate number of frames for this segment
            # Include both start and end frames
            num_frames = kf_next.frame_number - kf_curr.frame_number + 1

            # Adjust to 4n+1 format for FLF2V model requirement
            # Valid values: 5, 9, 13, 17, 21, 25, ..., 81, 85, etc.
            original_frames = num_frames
            if (num_frames - 1) % 4 != 0:
                # Find nearest valid frame count
                # Round to nearest 4n+1
                n = round((num_frames - 1) / 4)
                num_frames = 4 * n + 1

                # Ensure minimum of 5 frames
                if num_frames < 5:
                    num_frames = 5

                logger.warning(
                    f"Segment {i}: Adjusted frame count from {original_frames} to {num_frames} "
                    f"to match 4n+1 format required by FLF2V"
                )

            # Get segment-specific prompt if available
            sub_prompt = None
            if sub_prompts and i < len(sub_prompts):
                sub_prompt = sub_prompts[i]

            segment = SegmentConfig(
                segment_id=i,
                first_frame=kf_curr.frame_path,
                last_frame=kf_next.frame_path,
                num_frames=num_frames,
                prompt=global_prompt,
                sub_prompt=sub_prompt
            )

            segments.append(segment)

            logger.info(
                f"Segment {i}: {kf_curr.timestamp_sec:.2f}s -> {kf_next.timestamp_sec:.2f}s "
                f"({num_frames} frames)"
            )

        return segments

    def suggest_frame_adjustments(
        self,
        min_frames: int = 32,
        max_frames: int = 100,
        preferred_multiple: int = 4  # For 4n+1 format
    ) -> List[Dict]:
        """
        Analyze segments and suggest frame count adjustments.

        Args:
            min_frames: Minimum recommended frames per segment
            max_frames: Maximum recommended frames per segment
            preferred_multiple: Preferred frame count pattern (e.g., 4n+1)

        Returns:
            List of suggestions for each segment
        """
        suggestions = []

        for i in range(len(self.keyframes) - 1):
            kf_curr = self.keyframes[i]
            kf_next = self.keyframes[i + 1]
            original_frames = kf_next.frame_number - kf_curr.frame_number + 1

            suggestion = {
                "segment_id": i,
                "original_frames": original_frames,
                "original_duration": (kf_next.timestamp_sec - kf_curr.timestamp_sec),
                "status": "ok",
                "warnings": []
            }

            # Check if too short
            if original_frames < min_frames:
                suggestion["status"] = "warning"
                suggestion["warnings"].append(
                    f"Segment is short ({original_frames} frames). "
                    f"Recommended minimum: {min_frames} frames. "
                    f"May cause stiff motion."
                )
                suggestion["recommended_frames"] = min_frames
                suggestion["recommended_duration"] = min_frames / self.fps

            # Check if too long
            elif original_frames > max_frames:
                suggestion["status"] = "warning"
                suggestion["warnings"].append(
                    f"Segment is long ({original_frames} frames). "
                    f"Recommended maximum: {max_frames} frames. "
                    f"May cause concept drift."
                )
                suggestion["recommended_frames"] = max_frames
                suggestion["recommended_duration"] = max_frames / self.fps

            # Check if matches preferred pattern (4n+1)
            if (original_frames - 1) % preferred_multiple != 0:
                suggestion["warnings"].append(
                    f"Frame count ({original_frames}) doesn't match 4n+1 pattern. "
                    f"Consider adjusting for optimal performance."
                )
                # Suggest nearest 4n+1 value
                nearest = ((original_frames - 1) // preferred_multiple) * preferred_multiple + 1
                if nearest < original_frames:
                    nearest += preferred_multiple
                suggestion["recommended_frames_4n1"] = nearest

            suggestions.append(suggestion)

        return suggestions

    def print_summary(self):
        """Print a summary of the keyframe sequence"""
        print("\n" + "="*60)
        print("KEYFRAME SUMMARY")
        print("="*60)
        print(f"Total keyframes: {len(self.keyframes)}")
        print(f"FPS: {self.fps}")
        print(f"Total duration: {self.keyframes[-1].timestamp_sec:.2f} seconds")
        print(f"Total frames: {self.keyframes[-1].frame_number}")
        print()

        # Print keyframe details
        for i, kf in enumerate(self.keyframes):
            print(f"  Keyframe {i}: {kf.frame_path.name}")
            print(f"    Time: {kf.timestamp_sec:.2f}s (frame {kf.frame_number})")

        print()

        # Print segment info
        print(f"Segments to generate: {len(self.keyframes) - 1}")
        for i in range(len(self.keyframes) - 1):
            duration = self.keyframes[i+1].timestamp_sec - self.keyframes[i].timestamp_sec
            frames = self.keyframes[i+1].frame_number - self.keyframes[i].frame_number + 1
            print(f"  Segment {i}: {duration:.2f}s ({frames} frames)")

        print("="*60 + "\n")
