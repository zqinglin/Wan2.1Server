# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Video Stitching Module

Handles concatenation of video segments using FFmpeg with support for
different stitching methods and encoding options.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class VideoStitcher:
    """
    Video stitcher for combining multiple video segments into a single file.

    Supports different stitching methods:
    - concat: Simple concatenation (fastest, no transition)
    - blend: Cross-fade between segments (experimental)
    """

    def __init__(
        self,
        fps: int = 24,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium"
    ):
        """
        Initialize VideoStitcher.

        Args:
            fps: Target frames per second
            codec: Video codec (libx264, libx265, etc.)
            crf: Constant Rate Factor (quality, 0-51, lower is better)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        """
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset

        # Check if ffmpeg is available
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                capture_output=True
            )
            logger.info("FFmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found! Please install FFmpeg.")
            raise RuntimeError("FFmpeg is required but not found in PATH")

    def stitch_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        method: str = "concat",
        remove_intermediates: bool = False,
        **method_kwargs
    ) -> Path:
        """
        Stitch multiple video segments into a single video.

        Args:
            video_paths: List of video file paths to stitch
            output_path: Output video path
            method: Stitching method ('concat' or 'blend')
            remove_intermediates: Remove intermediate files after stitching
            **method_kwargs: Additional arguments for specific methods

        Returns:
            Path to the stitched video file
        """
        if not video_paths:
            raise ValueError("No video paths provided")

        if len(video_paths) == 1:
            logger.warning("Only one video provided, copying directly")
            import shutil
            shutil.copy(video_paths[0], output_path)
            return output_path

        # Ensure all input files exist
        for path in video_paths:
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")

        logger.info(f"Stitching {len(video_paths)} videos using '{method}' method")

        if method == "concat":
            return self._concat_videos(
                video_paths,
                output_path,
                remove_intermediates
            )
        elif method == "blend":
            return self._blend_videos(
                video_paths,
                output_path,
                **method_kwargs
            )
        else:
            raise ValueError(f"Unknown stitching method: {method}")

    def _concat_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        remove_intermediates: bool = False
    ) -> Path:
        """
        Concatenate videos using FFmpeg concat demuxer.

        This is the fastest method but provides no transition between segments.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create concat list file
        concat_list_file = output_path.parent / "concat_list.txt"
        with open(concat_list_file, 'w') as f:
            for video_path in video_paths:
                # Use absolute paths to avoid issues
                f.write(f"file '{video_path.resolve()}'\n")

        logger.info(f"Created concat list: {concat_list_file}")

        try:
            # Step 1: Concatenate without re-encoding (fastest)
            output_raw = output_path.parent / (output_path.stem + "_raw.mp4")

            logger.info("Step 1/2: Concatenating segments (copy mode)...")
            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                str(output_raw)
            ]

            result = subprocess.run(
                cmd_concat,
                check=True,
                capture_output=True,
                text=True
            )

            if not output_raw.exists():
                raise RuntimeError("Concatenation failed: output file not created")

            logger.info(f"  ✓ Raw concatenation complete: {output_raw}")

            # Step 2: Re-encode for compatibility and consistent quality
            logger.info("Step 2/2: Re-encoding for compatibility...")
            cmd_encode = [
                "ffmpeg", "-y",
                "-i", str(output_raw),
                "-vf", f"fps={self.fps}",
                "-c:v", self.codec,
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]

            result = subprocess.run(
                cmd_encode,
                check=True,
                capture_output=True,
                text=True
            )

            if not output_path.exists():
                raise RuntimeError("Re-encoding failed: output file not created")

            logger.info(f"  ✓ Re-encoding complete: {output_path}")

            # Clean up intermediate files
            if remove_intermediates:
                logger.info("Cleaning up intermediate files...")
                concat_list_file.unlink()
                output_raw.unlink()
                logger.info("  ✓ Cleanup complete")
            else:
                # Still remove concat list but keep raw
                concat_list_file.unlink()

            # Get output file size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Stitching complete: {output_path} ({size_mb:.2f} MB)")

            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Video stitching failed: {e.stderr}")

    def _blend_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        blend_duration: float = 0.5,
        transition_type: str = "fade"
    ) -> Path:
        """
        Stitch videos with cross-fade transitions.

        Note: This is experimental and may not work perfectly with all videos.

        Args:
            video_paths: List of video paths
            output_path: Output path
            blend_duration: Duration of blend transition in seconds
            transition_type: Type of transition (fade, wipeleft, wiperight, etc.)
        """
        logger.warning("Blend mode is experimental and may have issues")

        if len(video_paths) == 2:
            # Simple two-video blend
            return self._blend_two_videos(
                video_paths[0],
                video_paths[1],
                output_path,
                blend_duration,
                transition_type
            )
        else:
            # For multiple videos, we need to build a complex filter chain
            # For now, fall back to concat with a warning
            logger.warning(
                f"Blend mode for {len(video_paths)} videos not fully implemented. "
                f"Falling back to concat mode."
            )
            return self._concat_videos(video_paths, output_path, False)

    def _blend_two_videos(
        self,
        video1: Path,
        video2: Path,
        output_path: Path,
        blend_duration: float,
        transition_type: str
    ) -> Path:
        """Blend two videos with a transition"""
        # Get duration of first video
        duration_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video1)
        ]

        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration1 = float(result.stdout.strip())

        # Calculate transition offset
        offset = duration1 - blend_duration

        logger.info(f"Blending with {transition_type} transition ({blend_duration}s)")
        logger.info(f"  Transition offset: {offset:.2f}s")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video1),
            "-i", str(video2),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition={transition_type}:duration={blend_duration}:offset={offset}[outv]",
            "-map", "[outv]",
            "-c:v", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Blend complete: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Blend failed: {e.stderr}")
            raise RuntimeError(f"Video blending failed: {e.stderr}")

    def get_video_info(self, video_path: Path) -> dict:
        """
        Get information about a video file.

        Returns:
            Dictionary with video metadata
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
            "-of", "json",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)

        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "24/1")
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 24

            return {
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": fps,
                "duration": float(stream.get("duration", 0)),
                "frames": int(stream.get("nb_frames", 0))
            }

        return {}
