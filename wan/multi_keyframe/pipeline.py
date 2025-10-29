# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Multi-Keyframe Pipeline

Main orchestration pipeline for multi-keyframe video generation.
Coordinates keyframe management, segment generation, stitching, and smoothing.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .keyframe_manager import KeyframeManager, GenerationConfig
from .segment_generator import FLF2VSegmentGenerator, ParallelFLF2VGenerator
from .video_stitcher import VideoStitcher
from .video_smoother import VideoSmoother

logger = logging.getLogger(__name__)


class MultiKeyframePipeline:
    """
    Complete pipeline for multi-keyframe video generation.

    This pipeline orchestrates the entire process:
    1. Keyframe preparation and validation
    2. Segment generation (sequential or parallel)
    3. Video stitching
    4. Post-processing smoothing (optional)
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize the pipeline.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.keyframe_manager: Optional[KeyframeManager] = None
        self.generator: Optional[FLF2VSegmentGenerator] = None
        self.stitcher = VideoStitcher(fps=config.fps)
        self.smoother: Optional[VideoSmoother] = None

        # Validate configuration
        is_valid, message = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {message}")

        logger.info("="*60)
        logger.info("Multi-Keyframe Video Generation Pipeline")
        logger.info("="*60)
        logger.info(f"Model: {config.ckpt_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Resolution: {config.size}")
        logger.info(f"FPS: {config.fps}")
        logger.info("="*60)

    def run(
        self,
        keyframes: List[Tuple[str, float]],
        global_prompt: str,
        sub_prompts: Optional[List[str]] = None,
        output_path: str = "./output_final.mp4",
        num_gpus: int = 1,
        gpu_ids: Optional[List[int]] = None,
        enable_smoothing: bool = True,
        smoothing_method: str = "temporal_filter",
        force_regenerate: bool = False,
        resume_from: Optional[int] = None,
        keep_intermediates: bool = False
    ) -> Path:
        """
        Execute the complete pipeline.

        Args:
            keyframes: List of (frame_path, timestamp_sec) tuples
            global_prompt: Main prompt for all segments
            sub_prompts: Optional segment-specific prompts
            output_path: Final output video path
            num_gpus: Number of GPUs for parallel generation
            gpu_ids: Specific GPU IDs to use
            enable_smoothing: Enable post-processing smoothing
            smoothing_method: Smoothing method to use
            force_regenerate: Force regeneration of existing segments
            resume_from: Resume from specific segment ID
            keep_intermediates: Keep intermediate files

        Returns:
            Path to final generated video
        """
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Keyframe Preparation
        self._step1_prepare_keyframes(keyframes, global_prompt, sub_prompts)

        # Step 2: Generate Segments
        video_paths = self._step2_generate_segments(
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            force_regenerate=force_regenerate,
            resume_from=resume_from
        )

        # Step 3: Stitch Videos
        stitched_path = self._step3_stitch_videos(video_paths, output_path)

        # Step 4: Post-processing Smoothing
        if enable_smoothing:
            final_path = self._step4_smooth_video(
                stitched_path,
                output_path,
                smoothing_method
            )
        else:
            final_path = output_path
            stitched_path.rename(final_path)
            logger.info("Smoothing disabled, using stitched video directly")

        # Cleanup
        if not keep_intermediates:
            self._cleanup_intermediates(stitched_path if enable_smoothing else None)

        # Final summary
        elapsed_time = time.time() - start_time
        self._print_summary(final_path, elapsed_time)

        return final_path

    def _step1_prepare_keyframes(
        self,
        keyframes: List[Tuple[str, float]],
        global_prompt: str,
        sub_prompts: Optional[List[str]] = None
    ):
        """Step 1: Prepare and validate keyframes"""
        print("\n" + "="*60)
        print("STEP 1: Keyframe Preparation")
        print("="*60)

        # Initialize keyframe manager
        self.keyframe_manager = KeyframeManager(keyframes, fps=self.config.fps)

        # Validate keyframes
        is_valid, message = self.keyframe_manager.validate()
        if not is_valid:
            raise ValueError(f"Keyframe validation failed: {message}")

        # Print summary
        self.keyframe_manager.print_summary()

        # Get segments
        self.segments = self.keyframe_manager.get_segments(global_prompt, sub_prompts)
        logger.info(f"Created {len(self.segments)} segment configurations")

        # Print prompt information
        logger.info("\nPrompt Configuration:")
        logger.info(f"  Global: {global_prompt}")
        if sub_prompts:
            for i, sp in enumerate(sub_prompts):
                logger.info(f"  Segment {i}: {sp}")

        # Check frame count recommendations
        suggestions = self.keyframe_manager.suggest_frame_adjustments()
        warnings_found = False

        for sug in suggestions:
            if sug["status"] == "warning":
                warnings_found = True
                logger.warning(f"Segment {sug['segment_id']}:")
                for warning in sug["warnings"]:
                    logger.warning(f"  {warning}")

        if warnings_found:
            logger.info(
                "\n⚠ Warnings found in segment configuration. "
                "Generation will proceed, but quality may be affected."
            )

        print("="*60 + "\n")

    def _step2_generate_segments(
        self,
        num_gpus: int = 1,
        gpu_ids: Optional[List[int]] = None,
        force_regenerate: bool = False,
        resume_from: Optional[int] = None
    ) -> List[Path]:
        """Step 2: Generate video segments"""
        print("\n" + "="*60)
        print("STEP 2: Segment Generation")
        print("="*60)

        logger.info(f"DEBUG: num_gpus parameter = {num_gpus}, type = {type(num_gpus)}")

        # Initialize generator
        if num_gpus > 1:
            self.generator = ParallelFLF2VGenerator(self.config)
            logger.info(f"Using parallel generation with {num_gpus} GPUs")

            if gpu_ids:
                logger.info(f"GPU IDs: {gpu_ids}")

            video_paths = self.generator.generate_batch_parallel(
                self.segments,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids
            )
        else:
            self.generator = FLF2VSegmentGenerator(self.config)
            logger.info("Using sequential generation (single GPU)")

            video_paths = self.generator.generate_batch_sequential(
                self.segments,
                resume_from=resume_from
            )

        logger.info(f"\n✓ All {len(video_paths)} segments generated successfully")
        print("="*60 + "\n")

        return video_paths

    def _step3_stitch_videos(
        self,
        video_paths: List[Path],
        output_path: Path
    ) -> Path:
        """Step 3: Stitch video segments"""
        print("\n" + "="*60)
        print("STEP 3: Video Stitching")
        print("="*60)

        stitched_path = output_path.parent / (output_path.stem + "_stitched.mp4")

        logger.info(f"Stitching {len(video_paths)} segments...")

        result_path = self.stitcher.stitch_videos(
            video_paths,
            stitched_path,
            method="concat",
            remove_intermediates=False
        )

        logger.info(f"✓ Stitching complete")
        print("="*60 + "\n")

        return result_path

    def _step4_smooth_video(
        self,
        input_path: Path,
        output_path: Path,
        smoothing_method: str
    ) -> Path:
        """Step 4: Apply post-processing smoothing"""
        print("\n" + "="*60)
        print("STEP 4: Post-Processing Smoothing")
        print("="*60)

        # Initialize smoother
        self.smoother = VideoSmoother(method=smoothing_method)

        # Calculate junction frame numbers
        junction_frames = []
        if self.keyframe_manager:
            cumulative_frames = 0
            for segment in self.segments[:-1]:
                cumulative_frames += segment.num_frames - 1  # -1 because frames overlap
                junction_frames.append(cumulative_frames)

            logger.info(f"Junction points: {junction_frames}")

        # Apply smoothing
        result_path = self.smoother.smooth_video(
            input_path,
            output_path,
            junction_frames=junction_frames
        )

        logger.info(f"✓ Smoothing complete")
        print("="*60 + "\n")

        return result_path

    def _cleanup_intermediates(self, stitched_path: Optional[Path] = None):
        """Clean up intermediate files"""
        logger.info("Keeping intermediate files for reference...")

        # Keep stitched version for comparison
        if stitched_path and stitched_path.exists():
            logger.info(f"  Kept: {stitched_path.name}")

        # Note: We keep segment files in segments/ directory for potential reuse

    def _print_summary(self, final_path: Path, elapsed_time: float):
        """Print final summary"""
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

        # File info
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"Output file: {final_path}")
        print(f"File size: {size_mb:.2f} MB")

        # Timing info
        minutes, seconds = divmod(int(elapsed_time), 60)
        print(f"Total time: {minutes}m {seconds}s")

        # Video info
        try:
            info = self.stitcher.get_video_info(final_path)
            if info:
                print(f"Resolution: {info['width']}x{info['height']}")
                print(f"FPS: {info['fps']:.2f}")
                print(f"Duration: {info['duration']:.2f}s")
                print(f"Frames: {info['frames']}")
        except Exception as e:
            logger.debug(f"Could not get video info: {e}")

        print("="*60 + "\n")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Dictionary with status information
        """
        status = {
            "config": {
                "ckpt_dir": str(self.config.ckpt_dir),
                "size": self.config.size,
                "fps": self.config.fps,
                "output_dir": str(self.config.output_dir),
            },
            "segments": None,
            "completed_segments": [],
            "failed_segments": [],
        }

        if self.keyframe_manager:
            status["segments"] = {
                "total": len(self.segments),
                "details": [
                    {
                        "id": seg.segment_id,
                        "frames": seg.num_frames,
                        "first_frame": seg.first_frame.name,
                        "last_frame": seg.last_frame.name,
                    }
                    for seg in self.segments
                ]
            }

        if self.generator:
            status["completed_segments"] = self.generator.get_completed_segments()
            status["failed_segments"] = self.generator.get_failed_segments()

        return status
