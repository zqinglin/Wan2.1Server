# Keyframe Extraction Tool

快速从视频中提取均匀分布的关键帧，用于多关键帧视频生成测试。

## 快速使用

### 基础用法（提取4帧）

```bash
python examples/extract_keyframes.py dance.mp4 \
  --num_frames 5 \
  --output_dir examples/dance_keyframes_5frames
```

### 完整参数

```bash
python examples/extract_keyframes.py <video_path> \
  --num_frames 4 \              # 提取帧数
  --output_dir keyframes/ \     # 输出目录
  --height 720 \                # 目标高度（保持宽高比）
  --format png                  # 输出格式 (png/jpg)
```

## 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `video` | - | 必需 | 输入视频路径 |
| `--num_frames` | `-n` | 4 | 提取的关键帧数量（至少2帧） |
| `--output_dir` | `-o` | `keyframes` | 输出目录 |
| `--height` | - | 720 | 目标高度（像素，保持宽高比） |
| `--format` | `-f` | `png` | 输出格式（png/jpg） |

## 工作原理

1. **分析视频**: 使用 `ffprobe` 获取视频时长、帧率等信息
2. **计算时间戳**: 在视频时长内均匀分布 N 个时间点
3. **提取帧**: 使用 `ffmpeg` 在每个时间点提取一帧
4. **调整尺寸**: 自动缩放到指定高度（720p），保持宽高比
5. **生成配置**: 输出可用于 multi-keyframe 的配置片段

## 示例

### 示例 1: 从 dance.mp4 提取4帧用于测试

```bash
# 提取4帧到 dance_keyframes 目录
python examples/extract_keyframes.py Wan2.1/dance.mp4 \
  --num_frames 4 \
  --output_dir examples/dance_keyframes
```

输出：
```
examples/dance_keyframes/
├── frame_001.png  # t=0.0s
├── frame_002.png  # t=2.5s
├── frame_003.png  # t=5.0s
└── frame_004.png  # t=7.5s
```

### 示例 2: 提取更多帧（6帧）

```bash
python examples/extract_keyframes.py Wan2.1/dance.mp4 \
  --num_frames 6 \
  --output_dir examples/dance_keyframes_6
```

### 示例 3: 1080p 高清提取

```bash
python examples/extract_keyframes.py Wan2.1/dance.mp4 \
  --num_frames 4 \
  --height 1080 \
  --output_dir examples/dance_keyframes_1080p
```

### 示例 4: JPG 格式（更小文件）

```bash
python examples/extract_keyframes.py Wan2.1/dance.mp4 \
  --num_frames 4 \
  --format jpg \
  --output_dir examples/dance_keyframes_jpg
```

## 输出信息

脚本会自动输出：

1. **视频信息**: 分辨率、FPS、时长、总帧数
2. **提取的时间戳**: 每个关键帧的时间点
3. **验证结果**: 每个提取帧的尺寸
4. **配置片段**: 可直接复制到 config.yaml 或 CLI 参数

示例输出：
```
============================================================
KEYFRAME EXTRACTION
============================================================
Video: Wan2.1/dance.mp4
Output directory: examples/dance_keyframes
Number of frames: 4
Target height: 720p
============================================================

Analyzing video...
  Resolution: 1920x1080
  FPS: 30.00
  Duration: 10.00s
  Total frames: 300

Extracting 4 frames at timestamps:
  Frame 1: 0.00s
  Frame 2: 3.33s
  Frame 3: 6.67s
  Frame 4: 10.00s

Extracting frames...
✓ Extracted frame at 0.00s -> examples/dance_keyframes/frame_001.png
✓ Extracted frame at 3.33s -> examples/dance_keyframes/frame_002.png
✓ Extracted frame at 6.67s -> examples/dance_keyframes/frame_003.png
✓ Extracted frame at 10.00s -> examples/dance_keyframes/frame_004.png

Verifying extracted frames...
  Frame 1: 1280x720 - OK
  Frame 2: 1280x720 - OK
  Frame 3: 1280x720 - OK
  Frame 4: 1280x720 - OK

============================================================
EXTRACTION COMPLETE
============================================================
✓ 4 keyframes saved to: examples/dance_keyframes

📋 Add this to your config.yaml:
------------------------------------------------------------
keyframes:
  - ["examples/dance_keyframes/frame_001.png", 0.0]
  - ["examples/dance_keyframes/frame_002.png", 3.3]
  - ["examples/dance_keyframes/frame_003.png", 6.7]
  - ["examples/dance_keyframes/frame_004.png", 10.0]

# Or as CLI argument:
--keyframes "examples/dance_keyframes/frame_001.png:0.0,examples/dance_keyframes/frame_002.png:3.3,examples/dance_keyframes/frame_003.png:6.7,examples/dance_keyframes/frame_004.png:10.0"
============================================================
```

## 完整测试流程

### Step 1: 提取关键帧

```bash
python examples/extract_keyframes.py Wan2.1/dance.mp4 \
  --num_frames 4 \
  --output_dir examples/dance_keyframes
```

### Step 2: 创建测试配置

将输出的配置片段复制到 `examples/dance_test_config.yaml`:

```yaml
keyframes:
  - ["examples/dance_keyframes/frame_001.png", 0.0]
  - ["examples/dance_keyframes/frame_002.png", 3.3]
  - ["examples/dance_keyframes/frame_003.png", 6.7]
  - ["examples/dance_keyframes/frame_004.png", 10.0]

prompt: "Dancing performance, smooth movements, cinematic style, 4k quality"

sub_prompts:
  - "Dancer begins with elegant pose"
  - "Fluid transition to dynamic movement"
  - "Graceful spin and extension"

ckpt_dir: "/data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P"
size: "1280*720"
fps: 24
sample_steps: 50
sample_shift: 16.0
base_seed: 42
enable_smoothing: true
output: "examples/dance_reconstructed.mp4"
```

### Step 3: 运行多关键帧生成

```bash
python scripts/multi_keyframe_generate.py \
  --config examples/dance_test_config.yaml
```

### Step 4: 比较结果

```bash
# 原始视频
mpv Wan2.1/dance.mp4

# 重建视频
mpv examples/dance_reconstructed.mp4
```

## 技术细节

### 时间戳计算

对于 N 帧，时间戳计算公式：
```
timestamp[i] = (i / (N - 1)) * duration
```

示例（10秒视频，4帧）：
- Frame 0: (0/3) × 10 = 0.00s
- Frame 1: (1/3) × 10 = 3.33s
- Frame 2: (2/3) × 10 = 6.67s
- Frame 3: (3/3) × 10 = 10.00s

### 尺寸调整

使用 FFmpeg 的 `scale` 滤镜：
```
-vf scale=-2:720
```
- `-2`: 自动计算宽度以保持宽高比（确保是偶数）
- `720`: 目标高度

### 依赖

需要安装：
- **FFmpeg**: 视频处理
- **FFprobe**: 视频信息获取
- **Python Pillow**: 图像验证

检查安装：
```bash
ffmpeg -version
ffprobe -version
python -c "from PIL import Image; print('Pillow OK')"
```

## 常见问题

### Q: 为什么要均匀提取而不是手动选择？

A: 均匀提取适合快速测试和评估。对于正式创作，建议手动选择关键内容帧。

### Q: 可以提取非均匀分布的帧吗？

A: 可以！编辑脚本中的 `timestamps` 列表，或直接使用 FFmpeg 命令提取特定时间点的帧。

### Q: 提取的帧模糊或质量差？

A: 确保：
1. 源视频质量足够高
2. 使用 PNG 格式（无损）
3. 调整 `--height` 参数匹配源视频分辨率

### Q: 视频时长未知或不准确？

A: 某些视频格式可能无法准确获取时长。可以手动指定或使用其他工具预处理。

## 高级用法

### 自定义时间戳提取

编辑脚本中的时间戳生成部分：

```python
# 原始（均匀分布）
timestamps = [(i / (num_frames - 1)) * duration for i in range(num_frames)]

# 自定义（例如：指数分布）
import numpy as np
timestamps = np.linspace(0, duration, num_frames) ** 1.5

# 手动指定
timestamps = [0.0, 2.5, 7.8, 10.0]
```

### 批量处理多个视频

```bash
#!/bin/bash
for video in videos/*.mp4; do
    basename=$(basename "$video" .mp4)
    python examples/extract_keyframes.py "$video" \
        --num_frames 4 \
        --output_dir "keyframes/$basename"
done
```

## 脚本位置

- 脚本: `examples/extract_keyframes.py`
- 使用说明: `examples/EXTRACT_KEYFRAMES_README.md`

---

Happy frame extraction! 🎬✂️
