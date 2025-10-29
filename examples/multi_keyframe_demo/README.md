# Multi-Keyframe Video Generation Example

This example demonstrates how to generate long-form videos from multiple keyframes using the Wan2.1 FLF2V model.

## Quick Start

### 1. Prepare Your Keyframes

Place your keyframe images in the `keyframes/` directory. Keyframes should:
- Be in the same resolution
- Be in supported formats (PNG, JPG, etc.)
- Represent key moments in your desired video

Example structure:
```
keyframes/
├── frame_001.png  # t=0.0s
├── frame_002.png  # t=2.5s
├── frame_003.png  # t=5.0s
└── frame_004.png  # t=8.0s
```

### 2. Configure Your Generation

Edit `config.yaml` to set:
- Keyframe paths and timestamps
- Global and segment-specific prompts
- Model checkpoint path
- Generation parameters
- Output settings

### 3. Run Generation

#### Option A: Using Config File (Recommended)
```bash
cd /home/fanyijia/Wan2.1/Wan2.1
python scripts/multi_keyframe_generate.py --config examples/multi_keyframe_demo/config.yaml
```

#### Option B: Using Command Line Arguments
```bash
python scripts/multi_keyframe_generate.py \
  --keyframes "keyframes/frame_001.png:0.0,keyframes/frame_002.png:3.0,keyframes/frame_003.png:6.0" \
  --prompt "cinematic style, soft lighting, natural motion, 4k" \
  --ckpt_dir /data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P \
  --size 1280*720 \
  --output my_video.mp4
```

## Configuration Options

### Keyframes
- **Format**: `[path, timestamp_in_seconds]`
- **Timestamps**: Must be in chronological order
- **Minimum**: 2 keyframes required

### Prompts
- **Global prompt**: Describes overall style and quality
- **Sub-prompts**: Describe specific actions/changes in each segment

### Generation Parameters
- **sample_steps**: Higher = better quality but slower (default: 50)
- **sample_shift**: Controls noise schedule (default: 16.0 for FLF2V)
- **sample_guide_scale**: Classifier-free guidance strength (default: 5.0)
- **base_seed**: For reproducibility

### Performance
- **num_gpus**: Use multiple GPUs for parallel generation
- **offload_model**: Reduce VRAM usage (slower)
- **t5_cpu**: Keep text encoder on CPU

### Post-Processing
- **enable_smoothing**: Reduce flickering at segment junctions
- **smoothing_method**:
  - `temporal_filter`: FFmpeg temporal averaging (fast)
  - `optical_flow`: Optical flow smoothing (future)
  - `none`: No smoothing

## Advanced Usage

### Parallel Generation
Generate segments in parallel using multiple GPUs:

```bash
python scripts/multi_keyframe_generate.py \
  --config config.yaml \
  --num_gpus 4 \
  --gpu_ids 0,1,2,3
```

### Resume Interrupted Generation
If generation is interrupted, resume from a specific segment:

```bash
python scripts/multi_keyframe_generate.py \
  --config config.yaml \
  --resume_from 2  # Resume from segment 2
```

### Force Regeneration
Regenerate all segments even if they already exist:

```bash
python scripts/multi_keyframe_generate.py \
  --config config.yaml \
  --force_regenerate
```

## Tips for Best Results

### Segment Duration
- **Recommended**: 1.5 - 4 seconds per segment (36-96 frames @ 24fps)
- **Too short** (<32 frames): May cause stiff motion
- **Too long** (>100 frames): May cause concept drift

### Prompt Engineering
1. **Be specific**: Include details about style, lighting, camera movement
2. **Be consistent**: Keep character/scene descriptions consistent across segments
3. **Use sub-prompts**: Describe specific actions in each segment

Example:
```yaml
prompt: "Cinematic style, 4k quality, smooth camera movement"
sub_prompts:
  - "Camera pans across a misty forest at dawn"
  - "Zoom into a meadow with butterflies flying"
  - "Slow tilt up to reveal mountain peak at sunset"
```

### Frame Count Optimization
For best quality, aim for frame counts following the pattern `4n+1`:
- ✅ Good: 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81
- ⚠️ Acceptable: Other values work but may be suboptimal

### Content Consistency
To maintain consistent characters/scenes:
1. Use the **same seed** across runs
2. Include detailed descriptions in the **global prompt**
3. Ensure keyframes show the **same subjects**
4. Use **shorter segments** for complex scenes

## Troubleshooting

### Out of Memory (OOM)
```bash
# Option 1: Use model offloading
--offload_model --t5_cpu

# Option 2: Reduce resolution
--size 960*540

# Option 3: Use sequential generation
--num_gpus 1
```

### Flickering at Junctions
```bash
# Enable smoothing
--enable_smoothing --smoothing_method temporal_filter
```

### Inconsistent Content
- Strengthen global prompt with character/scene details
- Reduce segment length
- Increase sample steps (e.g., 60)

### Slow Generation
```bash
# Use parallel generation
--num_gpus 4

# Reduce quality settings
--sample_steps 40
```

## Output Files

After generation, you'll find:
- `output_final.mp4`: Final smoothed video
- `outputs/segments/`: Individual segment videos
- `outputs/segments/progress.json`: Generation progress

## Python API Usage

You can also use the pipeline programmatically:

```python
from pathlib import Path
from wan.multi_keyframe import MultiKeyframePipeline, GenerationConfig

# Configure
config = GenerationConfig(
    ckpt_dir=Path("/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"),
    size="1280*720",
    fps=24,
    base_seed=42
)

# Define keyframes
keyframes = [
    ("keyframes/frame_001.png", 0.0),
    ("keyframes/frame_002.png", 3.0),
    ("keyframes/frame_003.png", 6.0)
]

# Run pipeline
pipeline = MultiKeyframePipeline(config)
output = pipeline.run(
    keyframes=keyframes,
    global_prompt="Cinematic journey, 4k, smooth motion",
    output_path="my_video.mp4"
)
```

## Support

For issues or questions:
- Check the main repository README
- Review the implementation plan: `multi_keyframe_implementation_plan.md`
- Check logs with `--verbose` flag
