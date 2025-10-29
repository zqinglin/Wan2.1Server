# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Segment Generator Module

Handles individual video segment generation using the FLF2V model,
with support for sequential and parallel processing.
"""

import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .keyframe_manager import SegmentConfig, GenerationConfig

logger = logging.getLogger(__name__)


class FLF2VSegmentGenerator:
    """
    Generator for individual video segments using FLF2V model.

    This class wraps the generate.py script to create video segments
    from first and last frame pairs.
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize the segment generator.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.segments_dir = config.output_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.segments_dir / "progress.json"
        self.progress = self._load_progress()

        logger.info(f"Initialized FLF2VSegmentGenerator")
        logger.info(f"  Checkpoint: {config.ckpt_dir}")
        logger.info(f"  Output dir: {self.segments_dir}")

    def _load_progress(self) -> Dict[str, Any]:
        """Load generation progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed_segments": {}, "failed_segments": {}}

    def _save_progress(self, segment_id: int, output_path: Path, status: str = "completed"):
        """Save generation progress"""
        if status == "completed":
            self.progress["completed_segments"][str(segment_id)] = str(output_path)
        else:
            self.progress["failed_segments"][str(segment_id)] = str(output_path)

        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def generate_segment(
        self,
        segment: SegmentConfig,
        override_params: Optional[Dict[str, Any]] = None,
        force_regenerate: bool = False
    ) -> Path:
        """
        Generate a single video segment.

        Args:
            segment: Segment configuration
            override_params: Optional parameters to override config
            force_regenerate: If True, regenerate even if already exists

        Returns:
            Path to generated video file
        """
        output_path = self.segments_dir / f"segment_{segment.segment_id:03d}.mp4"

        # Check if already generated
        if not force_regenerate and str(segment.segment_id) in self.progress["completed_segments"]:
            existing_path = Path(self.progress["completed_segments"][str(segment.segment_id)])
            if existing_path.exists():
                logger.info(f"✓ Segment {segment.segment_id} already exists, skipping...")
                return existing_path

        # Build command
        cmd = self._build_command(segment, output_path, override_params)

        logger.info(f"[Segment {segment.segment_id}] Generating...")
        logger.info(f"  Duration: {segment.num_frames} frames")
        logger.info(f"  First frame: {segment.first_frame.name}")
        logger.info(f"  Last frame: {segment.last_frame.name}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Execute generation
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent  # Run from project root
                )

                # Log output if in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"STDOUT:\n{result.stdout}")

                if not output_path.exists():
                    raise RuntimeError(f"Generation completed but output file not found: {output_path}")

                logger.info(f"[Segment {segment.segment_id}] ✓ Completed: {output_path}")

                # Save progress
                self._save_progress(segment.segment_id, output_path, "completed")

                return output_path

            except subprocess.CalledProcessError as e:
                stderr = e.stderr if e.stderr else ""

                # Check for CUDA ECC errors
                is_cuda_ecc_error = "cudaErrorECCUncorrectable" in stderr or "CUDA error: uncorrectable ECC error" in stderr

                if is_cuda_ecc_error and attempt < max_retries - 1:
                    logger.warning(f"[Segment {segment.segment_id}] CUDA ECC error detected (attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"  Retrying after 5 seconds...")
                    import time
                    time.sleep(5)
                    continue

                logger.error(f"[Segment {segment.segment_id}] ✗ Generation failed")
                logger.error(f"  Error: {stderr}")
                self._save_progress(segment.segment_id, output_path, "failed")
                raise RuntimeError(f"Segment {segment.segment_id} generation failed: {stderr}")

    def _build_command(
        self,
        segment: SegmentConfig,
        output_path: Path,
        override_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Build the generation command"""
        cmd = [
            sys.executable,  # Use current Python interpreter
            "generate.py",
            "--task", "flf2v-14B",
            "--ckpt_dir", str(self.config.ckpt_dir),
            "--first_frame", str(segment.first_frame),
            "--last_frame", str(segment.last_frame),
            "--frame_num", str(segment.num_frames),
            "--prompt", segment.full_prompt,
            "--size", self.config.size,
            "--save_file", str(output_path),
        ]

        # Add optional parameters from config
        if self.config.sample_steps is not None:
            cmd.extend(["--sample_steps", str(self.config.sample_steps)])

        if self.config.sample_shift is not None:
            cmd.extend(["--sample_shift", str(self.config.sample_shift)])

        if self.config.sample_guide_scale is not None:
            cmd.extend(["--sample_guide_scale", str(self.config.sample_guide_scale)])

        if self.config.base_seed >= 0:
            # Use different seed for each segment
            segment_seed = self.config.base_seed + segment.segment_id
            cmd.extend(["--base_seed", str(segment_seed)])

        if self.config.offload_model:
            cmd.extend(["--offload_model", "True"])

        if self.config.t5_cpu:
            cmd.append("--t5_cpu")

        # Apply overrides
        if override_params:
            for key, value in override_params.items():
                # Remove existing parameter if present
                try:
                    idx = cmd.index(f"--{key}")
                    cmd.pop(idx)  # Remove key
                    cmd.pop(idx)  # Remove value
                except ValueError:
                    pass

                # Add new parameter
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])

        return cmd

    def generate_batch_sequential(
        self,
        segments: List[SegmentConfig],
        resume_from: Optional[int] = None,
        override_params: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """
        Generate all segments sequentially.

        Args:
            segments: List of segment configurations
            resume_from: Optional segment ID to resume from
            override_params: Optional parameters to override config

        Returns:
            List of paths to generated video files
        """
        outputs = []
        start_idx = resume_from if resume_from is not None else 0

        logger.info(f"Starting sequential generation of {len(segments)} segments")
        if start_idx > 0:
            logger.info(f"Resuming from segment {start_idx}")

        # Use tqdm if available
        segment_iter = segments[start_idx:]
        if HAS_TQDM:
            segment_iter = tqdm(
                segment_iter,
                desc="Generating segments",
                unit="segment",
                initial=start_idx,
                total=len(segments)
            )

        for segment in segment_iter:
            try:
                if HAS_TQDM:
                    segment_iter.set_description(f"Segment {segment.segment_id}")

                output_path = self.generate_segment(segment, override_params)
                outputs.append(output_path)

            except Exception as e:
                logger.error(f"Failed at segment {segment.segment_id}: {e}")
                logger.info(f"You can resume from this point using resume_from={segment.segment_id}")
                raise

        logger.info(f"✓ All {len(outputs)} segments generated successfully")
        return outputs

    def get_completed_segments(self) -> List[int]:
        """Get list of completed segment IDs"""
        return [int(sid) for sid in self.progress["completed_segments"].keys()]

    def get_failed_segments(self) -> List[int]:
        """Get list of failed segment IDs"""
        return [int(sid) for sid in self.progress["failed_segments"].keys()]

    def clear_progress(self):
        """Clear all progress tracking"""
        self.progress = {"completed_segments": {}, "failed_segments": {}}
        if self.progress_file.exists():
            self.progress_file.unlink()
        logger.info("Progress cleared")


class ParallelFLF2VGenerator(FLF2VSegmentGenerator):
    """
    Parallel version of FLF2VSegmentGenerator supporting multi-GPU generation.

    This generator can distribute segment generation across multiple GPUs
    for faster processing.
    """

    def generate_batch_parallel(
        self,
        segments: List[SegmentConfig],
        num_gpus: int = 1,
        gpu_ids: Optional[List[int]] = None,
        override_params: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """
        Generate all segments in parallel across multiple GPUs.

        Args:
            segments: List of segment configurations
            num_gpus: Number of GPUs to use
            gpu_ids: Specific GPU IDs to use (if None, uses 0 to num_gpus-1)
            override_params: Optional parameters to override config

        Returns:
            List of paths to generated video files
        """
        # Fall back to sequential if only 1 GPU
        if num_gpus == 1:
            logger.info("Single GPU mode, using sequential generation")
            return self.generate_batch_sequential(segments, override_params=override_params)

        if gpu_ids is None:
            gpu_ids = list(range(num_gpus))
        elif len(gpu_ids) != num_gpus:
            logger.warning(f"gpu_ids length ({len(gpu_ids)}) doesn't match num_gpus ({num_gpus})")
            gpu_ids = gpu_ids[:num_gpus]

        logger.info(f"Starting parallel generation using {num_gpus} GPUs: {gpu_ids}")

        # Distribute segments across GPUs
        segment_batches = [[] for _ in range(num_gpus)]
        for i, segment in enumerate(segments):
            gpu_idx = i % num_gpus
            segment_batches[gpu_idx].append((segment, gpu_ids[gpu_idx]))

        # Print distribution
        for gpu_idx, batch in enumerate(segment_batches):
            if batch:
                segment_ids = [s[0].segment_id for s in batch]
                logger.info(f"  GPU {gpu_ids[gpu_idx]}: segments {segment_ids}")

        # Track outputs
        outputs = [None] * len(segments)

        # Execute in parallel
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}

            # Submit all tasks
            for gpu_idx, batch in enumerate(segment_batches):
                for segment, gpu_id in batch:
                    future = executor.submit(
                        self._generate_on_gpu,
                        segment,
                        gpu_id,
                        override_params
                    )
                    futures[future] = segment.segment_id

            # Collect results as they complete
            completed = 0
            total = len(segments)

            # Use tqdm if available
            future_iter = as_completed(futures)
            if HAS_TQDM:
                future_iter = tqdm(
                    future_iter,
                    desc="Generating segments (parallel)",
                    total=total,
                    unit="segment"
                )

            for future in future_iter:
                segment_id = futures[future]
                try:
                    output_path = future.result()
                    outputs[segment_id] = output_path
                    completed += 1

                    if HAS_TQDM:
                        future_iter.set_postfix({"completed": f"{completed}/{total}"})
                    else:
                        logger.info(f"✓ Segment {segment_id} completed ({completed}/{total})")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"[Segment {segment_id}] ✗ Generation failed")
                    logger.error(f"  Error: {error_msg}")

                    # Check if it's a CUDA ECC error that might be recoverable
                    if "cudaErrorECCUncorrectable" in error_msg or "CUDA error: uncorrectable ECC error" in error_msg:
                        logger.error(f"  Hardware CUDA ECC error detected. This may require:")
                        logger.error(f"    1. GPU reset: nvidia-smi --gpu-reset")
                        logger.error(f"    2. Check GPU health: nvidia-smi -q | grep -i ECC")
                        logger.error(f"    3. Reduce num_gpus if GPU is failing")

                    raise

        logger.info(f"✓ All {len(outputs)} segments generated successfully")
        return outputs

    def _generate_on_gpu(
        self,
        segment: SegmentConfig,
        gpu_id: int,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate a segment on a specific GPU.

        This method is executed in a separate process.
        """
        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Generate segment
        return self.generate_segment(segment, override_params)
