# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Video Smoothing Module

Provides post-processing smoothing methods to reduce flickering and
jumps at segment junction points.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class VideoSmoother:
    """
    Video smoother for reducing artifacts at segment junctions.

    Supports multiple smoothing methods:
    - none: No smoothing (passthrough)
    - temporal_filter: FFmpeg temporal filtering
    - optical_flow: Optical flow-based smoothing (future implementation)
    """

    def __init__(self, method: str = "temporal_filter", junction_window: int = 3):
        """
        Initialize VideoSmoother.

        Args:
            method: Smoothing method to use
                - "none": No smoothing
                - "temporal_filter": Temporal averaging filter
                - "deflicker": Deflicker filter for brightness stabilization
                - "optical_flow": Optical flow interpolation (experimental)
            junction_window: Number of frames before/after junction to smooth (default: 3)
        """
        self.method = method
        self.junction_window = junction_window

        # Validate method
        valid_methods = ["none", "temporal_filter", "deflicker", "optical_flow"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid smoothing method: {method}. "
                f"Valid options: {valid_methods}"
            )

        logger.info(f"Initialized VideoSmoother with method: {method}, junction_window: {junction_window}")

    def smooth_video(
        self,
        input_path: Path,
        output_path: Path,
        junction_frames: Optional[List[int]] = None,
        **method_kwargs
    ) -> Path:
        """
        Apply smoothing to video.

        Args:
            input_path: Input video path
            output_path: Output video path
            junction_frames: List of frame numbers where segments are joined
            **method_kwargs: Additional arguments for specific methods

        Returns:
            Path to smoothed video
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying '{self.method}' smoothing")
        if junction_frames:
            logger.info(f"  Junction frames: {junction_frames}")

        if self.method == "none":
            return self._no_smoothing(input_path, output_path)
        elif self.method == "temporal_filter":
            return self._localized_smooth(
                input_path,
                output_path,
                junction_frames,
                filter_type="tmix",
                **method_kwargs
            )
        elif self.method == "deflicker":
            return self._localized_smooth(
                input_path,
                output_path,
                junction_frames,
                filter_type="deflicker",
                **method_kwargs
            )
        elif self.method == "optical_flow":
            return self._optical_flow_smooth(
                input_path,
                output_path,
                junction_frames,
                **method_kwargs
            )
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

    def _no_smoothing(self, input_path: Path, output_path: Path) -> Path:
        """Copy video without smoothing"""
        import shutil
        logger.info("No smoothing applied, copying file...")
        shutil.copy(input_path, output_path)
        logger.info(f"✓ Copy complete: {output_path}")
        return output_path

    def _temporal_filter_smooth(
        self,
        input_path: Path,
        output_path: Path,
        filter_type: str = "tmix",
        frames: int = 3,
        weights: str = "1 2 1"
    ) -> Path:
        """
        Apply temporal filtering using FFmpeg.

        This method averages frames over time to reduce flickering.

        Args:
            input_path: Input video
            output_path: Output video
            filter_type: Filter type ('tmix', 'tblend', 'minterpolate')
            frames: Number of frames to blend (for tmix)
            weights: Frame weights (for tmix)
        """
        logger.info(f"Applying temporal filter: {filter_type}")

        if filter_type == "tmix":
            # Temporal mix filter - averages consecutive frames
            vf = f"tmix=frames={frames}:weights='{weights}'"
        elif filter_type == "tblend":
            # Temporal blend - blends pairs of frames
            vf = "tblend=all_mode=average"
        elif filter_type == "minterpolate":
            # Motion interpolation
            vf = "minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        try:
            logger.info(f"  Filter: {vf}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if not output_path.exists():
                raise RuntimeError("Smoothing failed: output file not created")

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Temporal smoothing complete: {output_path} ({size_mb:.2f} MB)")

            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Video smoothing failed: {e.stderr}")

    def _localized_smooth(
        self,
        input_path: Path,
        output_path: Path,
        junction_frames: Optional[List[int]] = None,
        filter_type: str = "tmix",
        **filter_kwargs
    ) -> Path:
        """
        Apply smoothing only at junction points, preserving the rest of the video.

        Args:
            input_path: Input video
            output_path: Output video
            junction_frames: List of frame numbers where segments join
            filter_type: Type of filter ('tmix', 'deflicker', 'minterpolate')
            **filter_kwargs: Additional filter parameters

        Returns:
            Path to smoothed video
        """
        if not junction_frames or len(junction_frames) == 0:
            logger.info("No junction frames specified, applying global smoothing")
            return self._temporal_filter_smooth(
                input_path, output_path, filter_type=filter_type, **filter_kwargs
            )

        logger.info(f"Applying localized {filter_type} smoothing at {len(junction_frames)} junctions")
        logger.info(f"  Window size: ±{self.junction_window} frames around each junction")

        # Get video info to determine total frames
        import json
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "json",
            str(input_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        total_frames = int(info["streams"][0]["nb_read_frames"])

        logger.info(f"  Total frames in video: {total_frames}")

        # Build filter complex for localized smoothing
        segments = []
        filter_parts = []

        # Create smoothing filter based on type
        if filter_type == "tmix":
            frames = filter_kwargs.get("frames", 3)
            weights = filter_kwargs.get("weights", "1 2 1")
            smooth_filter = f"tmix=frames={frames}:weights='{weights}'"
        elif filter_type == "deflicker":
            mode = filter_kwargs.get("mode", "pm")
            size = filter_kwargs.get("size", 5)
            smooth_filter = f"deflicker=mode={mode}:size={size}"
        elif filter_type == "minterpolate":
            smooth_filter = "minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Split video into segments: [unsmoothed][smoothed][unsmoothed][smoothed]...
        last_end = 0
        segment_idx = 0

        for junction_frame in junction_frames:
            smooth_start = max(0, junction_frame - self.junction_window)
            smooth_end = min(total_frames - 1, junction_frame + self.junction_window)

            # Unsmoothed segment before junction
            if smooth_start > last_end:
                segments.append({
                    'start': last_end,
                    'end': smooth_start - 1,
                    'smooth': False,
                    'idx': segment_idx
                })
                segment_idx += 1

            # Smoothed segment at junction
            segments.append({
                'start': smooth_start,
                'end': smooth_end,
                'smooth': True,
                'idx': segment_idx
            })
            segment_idx += 1

            last_end = smooth_end + 1

        # Final unsmoothed segment
        if last_end < total_frames:
            segments.append({
                'start': last_end,
                'end': total_frames - 1,
                'smooth': False,
                'idx': segment_idx
            })

        logger.info(f"  Created {len(segments)} segments:")
        for seg in segments:
            seg_type = "SMOOTHED" if seg['smooth'] else "original"
            logger.info(f"    Segment {seg['idx']}: frames {seg['start']}-{seg['end']} ({seg_type})")

        # Build filter complex
        filter_complex = []
        concat_inputs = []

        for seg in segments:
            start_frame = seg['start']
            end_frame = seg['end']
            seg_idx = seg['idx']

            # Select frames for this segment
            select_filter = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=PTS-STARTPTS"

            if seg['smooth']:
                # Apply smoothing to this segment
                filter_complex.append(f"[0:v]{select_filter},{smooth_filter}[v{seg_idx}]")
            else:
                # Keep segment as-is
                filter_complex.append(f"[0:v]{select_filter}[v{seg_idx}]")

            concat_inputs.append(f"[v{seg_idx}]")

        # Concat all segments
        concat_filter = f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[outv]"
        filter_complex.append(concat_filter)

        filter_complex_str = ";".join(filter_complex)

        # Execute FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-filter_complex", filter_complex_str,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        try:
            logger.info(f"  Executing localized smoothing...")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if not output_path.exists():
                raise RuntimeError("Smoothing failed: output file not created")

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Localized smoothing complete: {output_path} ({size_mb:.2f} MB)")

            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Localized smoothing failed: {e.stderr}")

    def _optical_flow_smooth(
        self,
        input_path: Path,
        output_path: Path,
        junction_frames: Optional[List[int]] = None,
        window_size: int = 5
    ) -> Path:
        """
        Apply optical flow-based smoothing using minterpolate.

        Uses FFmpeg's motion interpolation filter for smoother transitions.
        """
        logger.info("Applying optical flow smoothing using minterpolate")

        return self._localized_smooth(
            input_path,
            output_path,
            junction_frames,
            filter_type="minterpolate"
        )

    def _extract_junction_regions(
        self,
        input_path: Path,
        junction_frames: List[int],
        window_size: int = 10
    ) -> List[Path]:
        """
        Extract video regions around junction points.

        This is useful for targeted smoothing only at junction points.

        Args:
            input_path: Input video
            junction_frames: List of junction frame numbers
            window_size: Number of frames before/after junction to extract

        Returns:
            List of paths to extracted video segments
        """
        extracted_segments = []

        for i, junction_frame in enumerate(junction_frames):
            start_frame = max(0, junction_frame - window_size)
            num_frames = window_size * 2 + 1

            output_segment = input_path.parent / f"junction_{i:03d}.mp4"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", f"select='between(n\\,{start_frame}\\,{start_frame + num_frames})',setpts=PTS-STARTPTS",
                "-c:v", "libx264",
                "-crf", "18",
                str(output_segment)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                extracted_segments.append(output_segment)
                logger.info(f"  Extracted junction {i}: {output_segment}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract junction {i}: {e.stderr}")

        return extracted_segments

    def apply_deflicker_filter(
        self,
        input_path: Path,
        output_path: Path,
        mode: str = "pm",  # 'pm' or 'am'
        size: int = 5,
        bypass: bool = False
    ) -> Path:
        """
        Apply FFmpeg's deflicker filter to reduce flickering.

        Args:
            input_path: Input video
            output_path: Output video
            mode: Filter mode ('pm' for pixel-wise, 'am' for average)
            size: Size of the temporal window
            bypass: Bypass mode (for comparison)

        Returns:
            Path to deflickered video
        """
        logger.info(f"Applying deflicker filter (mode={mode}, size={size})")

        vf = f"deflicker=mode={mode}:size={size}:bypass={1 if bypass else 0}"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            if not output_path.exists():
                raise RuntimeError("Deflicker failed: output file not created")

            logger.info(f"✓ Deflicker complete: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Deflicker failed: {e.stderr}")
            raise RuntimeError(f"Deflicker filter failed: {e.stderr}")
