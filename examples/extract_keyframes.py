#!/usr/bin/env python3
"""
Extract Keyframes from Video

This script extracts evenly-spaced keyframes from a video file
for use with the multi-keyframe generation pipeline.

Usage:
    python extract_keyframes.py <video_path> --num_frames 4 --output_dir keyframes/
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path

# Optional PIL import for verification
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def get_video_info(video_path):
    """Get video information using ffprobe"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "24/1")
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 24

            # Get duration
            duration = float(stream.get("duration", 0))

            # Get total frames (try multiple methods)
            total_frames = stream.get("nb_frames")
            if total_frames:
                total_frames = int(total_frames)
            else:
                # Estimate from duration and fps
                total_frames = int(duration * fps)

            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames
            }
        else:
            raise ValueError("No video stream found")

    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e.stderr}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error getting video info: {e}", file=sys.stderr)
        raise


def extract_frame(video_path, timestamp, output_path, target_height=720):
    """
    Extract a single frame at specified timestamp and resize to 720p

    Args:
        video_path: Path to input video
        timestamp: Time in seconds
        output_path: Path to save the frame
        target_height: Target height (maintains aspect ratio)
    """
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss", str(timestamp),  # Seek to timestamp
        "-i", str(video_path),
        "-vframes", "1",  # Extract 1 frame
        "-vf", f"scale=-2:{target_height}",  # Scale to 720p height, maintain aspect ratio
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Extracted frame at {timestamp:.2f}s -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to extract frame at {timestamp:.2f}s", file=sys.stderr)
        print(f"  Error: {e.stderr}", file=sys.stderr)
        raise


def extract_keyframes(
    video_path,
    num_frames=4,
    output_dir="keyframes",
    target_height=720,
    format="png"
):
    """
    Extract evenly-spaced keyframes from video

    Args:
        video_path: Path to input video
        num_frames: Number of keyframes to extract
        output_dir: Output directory for keyframes
        target_height: Target height in pixels
        format: Output format (png, jpg)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Validate input
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("KEYFRAME EXTRACTION")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of frames: {num_frames}")
    print(f"Target height: {target_height}p")
    print("="*60)

    # Get video info
    print("\nAnalyzing video...")
    info = get_video_info(video_path)

    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Total frames: {info['total_frames']}")

    # Calculate timestamps for evenly-spaced frames
    duration = info['duration']
    timestamps = []

    # Use slightly less than full duration to avoid seeking past end
    safe_duration = duration * 0.99  # Use 99% of duration for safety

    for i in range(num_frames):
        # Evenly space from 0 to safe_duration
        timestamp = (i / (num_frames - 1)) * safe_duration
        timestamps.append(timestamp)

    print(f"\nExtracting {num_frames} frames at timestamps:")
    for i, ts in enumerate(timestamps):
        print(f"  Frame {i+1}: {ts:.2f}s")

    print("\nExtracting frames...")

    # Extract frames
    keyframe_paths = []
    for i, timestamp in enumerate(timestamps):
        output_path = output_dir / f"frame_{i+1:03d}.{format}"
        extract_frame(video_path, timestamp, output_path, target_height)
        keyframe_paths.append(output_path)

    # Verify extracted frames
    print("\nVerifying extracted frames...")
    for i, path in enumerate(keyframe_paths):
        if path.exists():
            if HAS_PIL:
                with Image.open(path) as img:
                    print(f"  Frame {i+1}: {img.size[0]}x{img.size[1]} - OK")
            else:
                # Just check file exists
                size_kb = path.stat().st_size / 1024
                print(f"  Frame {i+1}: {size_kb:.1f} KB - OK")
        else:
            print(f"  Frame {i+1}: MISSING", file=sys.stderr)

    # Generate config snippet
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ {num_frames} keyframes saved to: {output_dir}")

    # Generate YAML config snippet
    print("\n📋 Add this to your config.yaml:")
    print("-"*60)
    print("keyframes:")
    for i, (path, ts) in enumerate(zip(keyframe_paths, timestamps)):
        print(f'  - ["{path}", {ts:.1f}]')

    print("\n# Or as CLI argument:")
    keyframes_str = ",".join([f"{path}:{ts:.1f}" for path, ts in zip(keyframe_paths, timestamps)])
    print(f'--keyframes "{keyframes_str}"')
    print("="*60)

    return keyframe_paths, timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Extract evenly-spaced keyframes from a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 4 keyframes at 720p
  python extract_keyframes.py dance.mp4 --num_frames 4 --output_dir keyframes/

  # Extract 6 keyframes at 1080p
  python extract_keyframes.py video.mp4 --num_frames 6 --height 1080

  # Extract as JPG instead of PNG
  python extract_keyframes.py video.mp4 --num_frames 4 --format jpg
        """
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )

    parser.add_argument(
        "--num_frames",
        "-n",
        type=int,
        default=4,
        help="Number of keyframes to extract (default: 4)"
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="keyframes",
        help="Output directory for keyframes (default: keyframes/)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target height in pixels (default: 720)"
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format (default: png)"
    )

    args = parser.parse_args()

    # Normalize format
    if args.format == "jpeg":
        args.format = "jpg"

    try:
        extract_keyframes(
            video_path=args.video,
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            target_height=args.height,
            format=args.format
        )
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
