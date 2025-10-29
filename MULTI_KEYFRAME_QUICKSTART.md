# Multi-Keyframe Video Generation - Quick Start Guide

🎬 Generate long-form videos from multiple keyframes using Wan2.1 FLF2V

## ✨ What's This?

This pipeline allows you to create seamless videos by:
1. Providing multiple keyframe images at different timestamps
2. Generating video segments between each pair of keyframes
3. Stitching segments together with optional smoothing

Perfect for: storyboards, animatics, creative transitions, long-form narratives

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Keyframes

Create a folder with your keyframe images:

```bash
mkdir my_keyframes
# Place your images: frame_001.png, frame_002.png, etc.
```

### Step 2: Run Generation

```bash
python scripts/multi_keyframe_generate.py \
  --keyframes "my_keyframes/frame_001.png:0.0,my_keyframes/frame_002.png:3.0,my_keyframes/frame_003.png:6.0" \
  --prompt "cinematic style, soft lighting, smooth motion, 4k" \
  --ckpt_dir /data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P \
  --output my_video.mp4
```

### Step 3: Enjoy!

Your video will be at `my_video.mp4` 🎉

---

## 📋 Recommended Configuration File Method

Create `config.yaml`:

```yaml
keyframes:
  - ["my_keyframes/frame_001.png", 0.0]
  - ["my_keyframes/frame_002.png", 3.0]
  - ["my_keyframes/frame_003.png", 6.0]

prompt: "cinematic style, soft lighting, smooth motion, 4k"

ckpt_dir: "/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"
size: "1280*720"
fps: 24
output: "my_video.mp4"
```

Run:
```bash
python scripts/multi_keyframe_generate.py --config config.yaml
```

---

## 🎯 Key Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--keyframes` | Frame paths and times | Required | `"f1.png:0,f2.png:3"` |
| `--prompt` | Global prompt | Required | `"cinematic, 4k"` |
| `--ckpt_dir` | Model checkpoint | Required | `/path/to/model` |
| `--size` | Resolution | `1280*720` | `1920*1080` |
| `--fps` | Frame rate | `24` | `30` |
| `--num_gpus` | Parallel GPUs | `1` | `4` |
| `--enable_smoothing` | Smooth junctions | `True` | - |
| `--base_seed` | Random seed | `-1` (random) | `42` |

---

## 💡 Tips for Best Results

### ✅ DO:
- Use **consistent keyframe sizes**
- Keep segments **1.5-4 seconds** long (36-96 frames)
- Write **detailed prompts** describing style and motion
- Use **sub-prompts** for segment-specific actions
- Enable **smoothing** to reduce flickering

### ❌ DON'T:
- Mix different resolutions in keyframes
- Create segments shorter than 32 frames (stiff motion)
- Create segments longer than 100 frames (concept drift)
- Skip important style details in prompts

### 🎨 Prompt Engineering Example

**Good:**
```yaml
prompt: "Cinematic documentary style, natural lighting, smooth camera movement, 4k quality"
sub_prompts:
  - "Slow pan across misty morning forest with soft golden light"
  - "Gentle zoom into flower meadow, butterflies entering frame"
  - "Smooth tilt up revealing mountain peak in warm sunset glow"
```

**Basic (works but less control):**
```yaml
prompt: "A journey through nature"
```

---

## 🔧 Advanced Usage

### Parallel Generation (Multiple GPUs)

```bash
python scripts/multi_keyframe_generate.py \
  --config config.yaml \
  --num_gpus 4 \
  --gpu_ids 0,1,2,3
```

### Resume Interrupted Generation

```bash
# If interrupted at segment 3, resume:
python scripts/multi_keyframe_generate.py \
  --config config.yaml \
  --resume_from 3
```

### Low VRAM Mode

```bash
python scripts/multi_keyframe_generate.py \
  --config config.yaml \
  --offload_model \
  --t5_cpu
```

---

## 📊 Pipeline Stages

```
┌─────────────────────────────────────────┐
│ STEP 1: Keyframe Preparation            │
│  • Validate keyframes                   │
│  • Calculate segment durations          │
│  • Check frame counts                   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ STEP 2: Segment Generation              │
│  • Generate each segment with FLF2V     │
│  • Support sequential or parallel       │
│  • Save progress for resume             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ STEP 3: Video Stitching                 │
│  • Concatenate segments with FFmpeg     │
│  • Re-encode for compatibility          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ STEP 4: Smoothing (Optional)            │
│  • Apply temporal filtering             │
│  • Reduce junction flickering           │
└─────────────────────────────────────────┘
              ↓
         Final Video! 🎬
```

---

## 🔍 Troubleshooting

### Problem: "Out of Memory (OOM)"

**Solution:**
```bash
--offload_model --t5_cpu  # Reduce VRAM usage
# OR
--size 960*540             # Lower resolution
# OR
--num_gpus 1               # Sequential (not parallel)
```

### Problem: Flickering at segment junctions

**Solution:**
```bash
--enable_smoothing --smoothing_method temporal_filter
```

### Problem: Content inconsistency across segments

**Solutions:**
1. Strengthen global prompt with character/scene details
2. Reduce segment length (shorter = more consistent)
3. Use same seed: `--base_seed 42`

### Problem: Generation too slow

**Solutions:**
```bash
--num_gpus 4               # Use multiple GPUs
# OR
--sample_steps 40          # Reduce quality slightly
```

---

## 📁 Output Files

After generation:

```
outputs/
├── segments/
│   ├── segment_000.mp4       # Individual segments
│   ├── segment_001.mp4
│   ├── segment_002.mp4
│   └── progress.json         # Progress tracking
├── output_stitched.mp4       # Before smoothing
└── output_final.mp4          # Final result ✨
```

---

## 📖 Example Configurations

### Example 1: Short Story (3 scenes)

```yaml
keyframes:
  - ["scene1_start.png", 0.0]
  - ["scene1_end.png", 2.0]
  - ["scene2_start.png", 2.0]
  - ["scene2_end.png", 4.5]
  - ["scene3_start.png", 4.5]
  - ["scene3_end.png", 7.0]

prompt: "Animated short film style, vibrant colors, smooth transitions"

sub_prompts:
  - "Character wakes up, stretches and yawns"
  - "Character walks to window, opens curtains"
  - "Bright sunlight floods room, character smiles"
  - "Character picks up backpack, walks to door"
  - "Character exits room, closes door behind"

ckpt_dir: "/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"
size: "1280*720"
fps: 24
base_seed: 42
enable_smoothing: true
```

### Example 2: Product Showcase (Rotating view)

```yaml
keyframes:
  - ["product_front.png", 0.0]
  - ["product_side.png", 1.5]
  - ["product_back.png", 3.0]
  - ["product_side2.png", 4.5]
  - ["product_front.png", 6.0]

prompt: "Professional product photography, studio lighting, 4k quality, smooth rotation"

ckpt_dir: "/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"
size: "1920*1080"
fps: 30
sample_steps: 60  # Higher quality
enable_smoothing: true
smoothing_method: "temporal_filter"
```

### Example 3: Landscape Timelapse

```yaml
keyframes:
  - ["landscape_dawn.png", 0.0]
  - ["landscape_morning.png", 3.0]
  - ["landscape_noon.png", 6.0]
  - ["landscape_afternoon.png", 9.0]
  - ["landscape_sunset.png", 12.0]
  - ["landscape_dusk.png", 15.0]

prompt: "Nature documentary, cinematic landscape, changing light conditions, 4k"

sub_prompts:
  - "Soft dawn light gradually illuminating the valley"
  - "Morning mist clearing, revealing details"
  - "Bright midday sun, sharp shadows"
  - "Warm afternoon light, gentle cloud movement"
  - "Golden hour glow spreading across landscape"

ckpt_dir: "/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"
size: "1920*1080"
fps: 24
num_gpus: 4  # Parallel generation for speed
```

---

## 🐍 Python API

Use programmatically in your scripts:

```python
from pathlib import Path
from wan.multi_keyframe import MultiKeyframePipeline, GenerationConfig

# Setup
config = GenerationConfig(
    ckpt_dir=Path("/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"),
    size="1280*720",
    fps=24,
    base_seed=42
)

# Define keyframes
keyframes = [
    ("keyframes/start.png", 0.0),
    ("keyframes/middle.png", 3.0),
    ("keyframes/end.png", 6.0)
]

# Generate
pipeline = MultiKeyframePipeline(config)
output = pipeline.run(
    keyframes=keyframes,
    global_prompt="Cinematic style, 4k, smooth motion",
    sub_prompts=[
        "Camera pans left across scene",
        "Camera zooms into subject"
    ],
    output_path="result.mp4",
    num_gpus=2,
    enable_smoothing=True
)

print(f"Video saved to: {output}")
```

---

## 📚 Further Reading

- **Full Implementation Plan**: `multi_keyframe_implementation_plan.md`
- **Example Configuration**: `examples/multi_keyframe_demo/config.yaml`
- **Detailed README**: `examples/multi_keyframe_demo/README.md`
- **Main Wan2.1 README**: `README.md`

---

## ⚡ Getting Help

1. **Check logs**: Add `--verbose` flag for detailed output
2. **Review examples**: See `examples/multi_keyframe_demo/`
3. **Common issues**: Check Troubleshooting section above
4. **Model issues**: Refer to main Wan2.1 documentation

---

**Happy Generating! 🎬✨**
