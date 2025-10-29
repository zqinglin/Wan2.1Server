# Dance Video Test - Multi-Keyframe Pipeline

这是一个使用 `dance.mp4` 测试多关键帧视频生成pipeline的完整示例。

## 📁 已准备的文件

```
examples/
├── dance_keyframes/              # 从dance.mp4提取的4个关键帧
│   ├── frame_001.png (0.0s)     - 1280x720, 534 KB
│   ├── frame_002.png (3.3s)     - 1280x720, 564 KB
│   ├── frame_003.png (6.6s)     - 1280x720, 422 KB
│   └── frame_004.png (9.9s)     - 1280x720, 240 KB
│
├── dance_test_config.yaml        # 测试配置文件
└── extract_keyframes.py          # 关键帧提取脚本
```

## 🚀 运行测试

### 方法1: 使用配置文件（推荐）

```bash
python scripts/multi_keyframe_generate.py \
  --config examples/dance_test_config.yaml
```

### 方法2: 命令行参数

```bash
python scripts/multi_keyframe_generate.py \
  --keyframes "examples/dance_keyframes/frame_001.png:0.0,examples/dance_keyframes/frame_002.png:3.3,examples/dance_keyframes/frame_003.png:6.6,examples/dance_keyframes/frame_004.png:9.9" \
  --prompt "Dance performance, smooth fluid movements, dynamic choreography, cinematic style, high quality, 4k" \
  --ckpt_dir /data1/huangwenlei/Code/Wan2.1-FLF2V-14B-720P \
  --size 1280*720 \
  --output examples/dance_reconstructed.mp4
```

### 方法3: 多GPU并行生成

```bash
# 编辑 dance_test_config.yaml，设置：
# num_gpus: 4

python scripts/multi_keyframe_generate.py \
  --config examples/dance_test_config.yaml
```

## 📊 预期结果

生成的视频应该：
- **分辨率**: 1280x720
- **帧率**: 24 FPS
- **时长**: ~10秒
- **片段**: 3个片段（4个关键帧 = 3个segment）
  - Segment 0: 0.0s → 3.3s (约79帧)
  - Segment 1: 3.3s → 6.6s (约79帧)
  - Segment 2: 6.6s → 9.9s (约79帧)

## 📝 生成过程

Pipeline会执行以下步骤：

```
STEP 1: Keyframe Preparation
  ✓ 验证4个关键帧
  ✓ 创建3个segment配置
  ✓ 检查帧数建议

STEP 2: Segment Generation
  ✓ 生成 segment_000.mp4 (frame_001 → frame_002)
  ✓ 生成 segment_001.mp4 (frame_002 → frame_003)
  ✓ 生成 segment_002.mp4 (frame_003 → frame_004)

STEP 3: Video Stitching
  ✓ 拼接3个片段
  ✓ 重新编码确保兼容性

STEP 4: Smoothing
  ✓ 应用temporal filter平滑
  ✓ 减少拼接处闪烁
```

## 🎬 输出文件

```
outputs/
└── segments/
    ├── segment_000.mp4           # 第1个片段
    ├── segment_001.mp4           # 第2个片段
    ├── segment_002.mp4           # 第3个片段
    └── progress.json             # 进度跟踪

examples/
└── dance_reconstructed.mp4       # 最终输出 ✨
```

## 🔍 比较原始视频

```bash
# 查看原始视频
mpv dance.mp4

# 查看重建视频
mpv examples/dance_reconstructed.mp4

# 并排比较（需要ffmpeg）
ffmpeg -i dance.mp4 -i examples/dance_reconstructed.mp4 \
  -filter_complex "[0:v][1:v]hstack[v]" \
  -map "[v]" -c:v libx264 -crf 18 comparison.mp4
```

## 🔧 调整参数

### 提高质量

编辑 `dance_test_config.yaml`:
```yaml
sample_steps: 60              # 增加采样步数
sample_guide_scale: 6.0       # 增加引导强度
```

### 加快速度

```yaml
sample_steps: 40              # 减少采样步数
num_gpus: 4                   # 使用多GPU并行
```

### 降低显存占用

```yaml
offload_model: true           # 模型卸载到CPU
t5_cpu: true                  # T5保持在CPU
```

## 📐 修改关键帧数量

重新提取不同数量的关键帧：

```bash
# 提取6个关键帧
python examples/extract_keyframes.py dance.mp4 \
  --num_frames 6 \
  --output_dir examples/dance_keyframes_6

# 使用新的关键帧生成
# 编辑配置文件或使用命令行
```

## 🐛 常见问题

### Q: 生成时间太长
**A**: 使用多GPU并行：`--num_gpus 4`

### Q: 显存不足 (OOM)
**A**: 启用模型卸载：`--offload_model --t5_cpu`

### Q: 拼接处有闪烁
**A**: 已启用smoothing，可尝试增加采样步数

### Q: 内容与原视频差异大
**A**: 这是正常的！模型在关键帧间"想象"过渡，不是简单复制。
可以：
1. 增加关键帧数量（更多约束）
2. 强化prompt描述
3. 增加sample_steps

## 📊 性能参考

基于 RTX 4090 + 14B 模型 + 720p：

| 配置 | 单片段时间 | 总时间（3片段） |
|------|-----------|---------------|
| 单GPU | ~4-5分钟 | ~12-15分钟 |
| 4 GPU并行 | ~4-5分钟 | ~5-6分钟 |
| offload模式 | ~7-8分钟 | ~21-24分钟 |

## 🎯 测试检查清单

- [ ] 关键帧成功提取（4个PNG文件）
- [ ] 配置文件路径正确
- [ ] 模型checkpoint路径正确
- [ ] 足够的磁盘空间（~2GB）
- [ ] 足够的GPU显存（~12GB）
- [ ] 成功生成3个片段
- [ ] 成功拼接视频
- [ ] 最终视频可以播放
- [ ] 视频分辨率正确（1280x720）
- [ ] 视频时长合理（~10秒）

## 💡 下一步

测试成功后，可以：

1. **尝试自己的视频**
   ```bash
   python examples/extract_keyframes.py your_video.mp4 -n 4
   ```

2. **调整提示词** 获得不同效果

3. **实验不同参数** 了解影响

4. **创建自定义关键帧** 而不是从视频提取

祝测试顺利！🎉
