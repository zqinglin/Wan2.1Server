#!/usr/bin/env python3
"""
Quick test script to verify multi-keyframe module installation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing multi-keyframe module imports...")
    print("="*60)

    try:
        from wan.multi_keyframe import (
            KeyframeInfo,
            SegmentConfig,
            GenerationConfig,
            KeyframeManager,
            FLF2VSegmentGenerator,
            ParallelFLF2VGenerator,
            VideoStitcher,
            VideoSmoother,
            MultiKeyframePipeline,
        )
        print("✓ All core classes imported successfully")

        # Test basic instantiation
        print("\nTesting basic instantiation...")

        # Test KeyframeManager
        keyframes = [
            ("test1.png", 0.0),
            ("test2.png", 2.0),
        ]
        km = KeyframeManager(keyframes, fps=24)
        print(f"✓ KeyframeManager: {len(km.keyframes)} keyframes")

        # Test GenerationConfig
        config = GenerationConfig(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            size="1280*720",
            fps=24
        )
        print(f"✓ GenerationConfig: {config.size} @ {config.fps}fps")

        # Test VideoStitcher
        stitcher = VideoStitcher(fps=24)
        print(f"✓ VideoStitcher: codec={stitcher.codec}")

        # Test VideoSmoother
        smoother = VideoSmoother(method="temporal_filter")
        print(f"✓ VideoSmoother: method={smoother.method}")

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_imports()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
