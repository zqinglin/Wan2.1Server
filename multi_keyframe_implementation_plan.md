# 多关键帧视频合成实现规划文档
## 基于 Wan2.1 FLF2V 的分段生成与拼接方案

---

## 一、项目概述

### 1.1 目标
实现一个基于 Wan2.1-FLF2V 模型的多关键帧视频合成工具，通过分段生成和智能拼接，支持从多个关键帧生成连贯的长视频。

### 1.2 核心优势
- ✅ **零训练工作流**：无需额外训练，直接使用现有 FLF2V 模型
- ✅ **易于并行化**：各片段独立生成，天然支持多GPU并行
- ✅ **灵活可控**：支持每段独立的提示词和参数配置
- ✅ **可扩展性强**：易于集成后处理和优化模块

---

## 二、实现架构

### 2.1 整体流程图

```
用户输入关键帧序列 + 全局配置
         ↓
[1] 关键帧准备与任务划分
         ↓
[2] 分段独立生成 (可并行)
         ↓
[3] 视频片段拼接
         ↓
[4] 后处理与平滑 (可选)
         ↓
    输出最终视频
```

### 2.2 核心模块设计

#### 模块 1: KeyframeManager (关键帧管理器)
**职责**：
- 验证关键帧序列的有效性
- 计算片段划分和帧数分配
- 管理关键帧元数据

**接口设计**：
```python
class KeyframeManager:
    def __init__(self, keyframes: List[KeyframeInfo], fps: int = 24):
        """
        Args:
            keyframes: [(frame_path, timestamp_sec), ...]
            fps: 目标视频帧率
        """

    def validate(self) -> bool:
        """验证关键帧序列：路径存在性、尺寸一致性、时间顺序"""

    def get_segments(self) -> List[SegmentConfig]:
        """返回所有片段的配置信息"""

    def suggest_frame_counts(self, min_frames=32, max_frames=100) -> List[int]:
        """根据时间间隔智能建议每段帧数"""
```

#### 模块 2: FLF2VSegmentGenerator (片段生成器)
**职责**：
- 封装对 generate.py 的调用
- 管理单个片段的生成参数
- 支持并行生成和错误恢复

**接口设计**：
```python
class FLF2VSegmentGenerator:
    def __init__(self, ckpt_dir: str, config: GenerationConfig):
        """初始化模型配置"""

    def generate_segment(
        self,
        first_frame: str,
        last_frame: str,
        num_frames: int,
        prompt: str,
        segment_id: int,
        **kwargs
    ) -> Path:
        """生成单个视频片段"""

    def generate_batch_parallel(
        self,
        segments: List[SegmentConfig],
        num_workers: int = 1
    ) -> List[Path]:
        """并行生成多个片段"""
```

#### 模块 3: VideoStitcher (视频拼接器)
**职责**：
- 使用 FFmpeg 拼接视频片段
- 处理编码参数和格式转换
- 可选：添加过渡效果

**接口设计**：
```python
class VideoStitcher:
    def __init__(self, fps: int = 24, codec: str = "libx264"):
        """初始化拼接参数"""

    def stitch_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        method: str = "concat"  # concat | blend
    ) -> Path:
        """拼接视频片段"""

    def add_transition(
        self,
        duration_sec: float = 0.5
    ):
        """在拼接点添加过渡效果（可选）"""
```

#### 模块 4: VideoSmoother (平滑处理器)
**职责**：
- 对拼接后的视频进行时域平滑
- 支持多种平滑方法

**接口设计**：
```python
class VideoSmoother:
    def __init__(self, method: str = "optical_flow"):
        """
        Args:
            method: "optical_flow" | "temporal_filter" | "rife" | "none"
        """

    def smooth_video(
        self,
        input_path: Path,
        output_path: Path,
        junction_frames: List[int]  # 拼接点帧号列表
    ) -> Path:
        """对视频进行平滑处理"""
```

---

## 三、详细实现步骤

### Phase 1: 基础框架搭建 (Week 1)

#### Step 1.1: 项目结构初始化
```
wan/
├── multi_keyframe/
│   ├── __init__.py
│   ├── keyframe_manager.py      # 关键帧管理
│   ├── segment_generator.py     # 片段生成
│   ├── video_stitcher.py        # 视频拼接
│   ├── video_smoother.py        # 平滑处理
│   └── pipeline.py              # 主流程编排
├── scripts/
│   └── multi_keyframe_generate.py  # 命令行工具
└── examples/
    └── multi_keyframe_demo/
        ├── keyframes/            # 示例关键帧
        └── config.yaml           # 配置文件示例
```

#### Step 1.2: 核心数据结构定义
```python
# keyframe_manager.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class KeyframeInfo:
    """关键帧信息"""
    frame_path: Path
    timestamp_sec: float
    frame_number: Optional[int] = None  # 自动计算

@dataclass
class SegmentConfig:
    """片段生成配置"""
    segment_id: int
    first_frame: Path
    last_frame: Path
    num_frames: int
    prompt: str
    sub_prompt: Optional[str] = None  # 局部提示词

@dataclass
class GenerationConfig:
    """全局生成配置"""
    ckpt_dir: Path
    size: str = "1280*720"
    fps: int = 24
    sample_steps: Optional[int] = None
    sample_shift: Optional[float] = None
    sample_guide_scale: float = 5.0
    base_seed: int = -1
    offload_model: bool = False
```

#### Step 1.3: KeyframeManager 实现
```python
class KeyframeManager:
    def __init__(
        self,
        keyframes: List[Tuple[str, float]],
        fps: int = 24
    ):
        self.fps = fps
        self.keyframes = [
            KeyframeInfo(
                frame_path=Path(path),
                timestamp_sec=ts,
                frame_number=int(ts * fps)
            )
            for path, ts in keyframes
        ]

    def validate(self) -> Tuple[bool, str]:
        """验证关键帧序列"""
        # 检查文件存在性
        for kf in self.keyframes:
            if not kf.frame_path.exists():
                return False, f"Frame not found: {kf.frame_path}"

        # 检查时间顺序
        timestamps = [kf.timestamp_sec for kf in self.keyframes]
        if timestamps != sorted(timestamps):
            return False, "Keyframes not in chronological order"

        # 检查图像尺寸一致性
        from PIL import Image
        sizes = [Image.open(kf.frame_path).size for kf in self.keyframes]
        if len(set(sizes)) > 1:
            return False, f"Inconsistent frame sizes: {set(sizes)}"

        return True, "Validation passed"

    def get_segments(
        self,
        global_prompt: str,
        sub_prompts: Optional[List[str]] = None
    ) -> List[SegmentConfig]:
        """生成片段配置"""
        segments = []

        for i in range(len(self.keyframes) - 1):
            kf_curr = self.keyframes[i]
            kf_next = self.keyframes[i + 1]

            num_frames = kf_next.frame_number - kf_curr.frame_number + 1

            # 合并全局和局部提示词
            segment_prompt = global_prompt
            if sub_prompts and i < len(sub_prompts):
                segment_prompt = f"{global_prompt}. {sub_prompts[i]}"

            segments.append(SegmentConfig(
                segment_id=i,
                first_frame=kf_curr.frame_path,
                last_frame=kf_next.frame_path,
                num_frames=num_frames,
                prompt=segment_prompt
            ))

        return segments

    def suggest_frame_adjustments(
        self,
        min_frames: int = 32,
        max_frames: int = 100,
        preferred_frames: int = 65  # 4n+1 格式
    ) -> List[Dict]:
        """建议帧数调整方案"""
        suggestions = []

        for i in range(len(self.keyframes) - 1):
            kf_curr = self.keyframes[i]
            kf_next = self.keyframes[i + 1]
            original_frames = kf_next.frame_number - kf_curr.frame_number + 1

            suggestion = {
                "segment_id": i,
                "original_frames": original_frames,
                "status": "ok"
            }

            if original_frames < min_frames:
                suggestion["status"] = "too_short"
                suggestion["recommended_frames"] = min_frames
                suggestion["warning"] = f"Segment too short ({original_frames} frames), may cause stiff motion"

            elif original_frames > max_frames:
                suggestion["status"] = "too_long"
                suggestion["recommended_frames"] = max_frames
                suggestion["warning"] = f"Segment too long ({original_frames} frames), may cause concept drift"

            suggestions.append(suggestion)

        return suggestions
```

---

### Phase 2: 片段生成实现 (Week 2)

#### Step 2.1: FLF2VSegmentGenerator 基础实现
```python
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json

class FLF2VSegmentGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.output_dir = Path("./segments_output")
        self.output_dir.mkdir(exist_ok=True)

    def generate_segment(
        self,
        segment: SegmentConfig,
        **override_kwargs
    ) -> Path:
        """生成单个片段"""
        output_path = self.output_dir / f"segment_{segment.segment_id:03d}.mp4"

        # 构建命令
        cmd = [
            "python", "generate.py",
            "--task", "flf2v-14B",
            "--ckpt_dir", str(self.config.ckpt_dir),
            "--first_frame", str(segment.first_frame),
            "--last_frame", str(segment.last_frame),
            "--frame_num", str(segment.num_frames),
            "--prompt", segment.prompt,
            "--size", self.config.size,
            "--fps", str(self.config.fps),
            "--save_file", str(output_path),
        ]

        # 添加可选参数
        if self.config.sample_steps:
            cmd.extend(["--sample_steps", str(self.config.sample_steps)])
        if self.config.sample_shift:
            cmd.extend(["--sample_shift", str(self.config.sample_shift)])
        if self.config.sample_guide_scale:
            cmd.extend(["--sample_guide_scale", str(self.config.sample_guide_scale)])
        if self.config.base_seed >= 0:
            cmd.extend(["--base_seed", str(self.config.base_seed)])
        if self.config.offload_model:
            cmd.extend(["--offload_model", "True"])

        # 应用覆盖参数
        for key, value in override_kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        print(f"[Segment {segment.segment_id}] Generating...")
        print(f"Command: {' '.join(cmd)}")

        # 执行生成
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[Segment {segment.segment_id}] ✓ Completed: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            print(f"[Segment {segment.segment_id}] ✗ Failed")
            print(f"Error: {e.stderr}")
            raise

    def generate_batch_sequential(
        self,
        segments: List[SegmentConfig],
        resume_from: Optional[int] = None
    ) -> List[Path]:
        """顺序生成所有片段（支持断点续传）"""
        outputs = []
        start_idx = resume_from if resume_from is not None else 0

        for segment in segments[start_idx:]:
            try:
                output_path = self.generate_segment(segment)
                outputs.append(output_path)

                # 保存进度
                self._save_progress(segment.segment_id, output_path)

            except Exception as e:
                print(f"Failed at segment {segment.segment_id}: {e}")
                print(f"You can resume from this point using --resume_from {segment.segment_id}")
                raise

        return outputs

    def _save_progress(self, segment_id: int, output_path: Path):
        """保存生成进度"""
        progress_file = self.output_dir / "progress.json"
        progress = {}

        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

        progress[str(segment_id)] = str(output_path)

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
```

#### Step 2.2: 并行生成支持
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
import torch

class ParallelFLF2VGenerator(FLF2VSegmentGenerator):
    """支持多GPU并行生成的版本"""

    def generate_batch_parallel(
        self,
        segments: List[SegmentConfig],
        num_gpus: int = 1,
        gpu_ids: Optional[List[int]] = None
    ) -> List[Path]:
        """并行生成（多GPU）"""

        if num_gpus == 1:
            return self.generate_batch_sequential(segments)

        if gpu_ids is None:
            gpu_ids = list(range(num_gpus))

        print(f"Parallel generation using {num_gpus} GPUs: {gpu_ids}")

        # 分配片段到GPU
        segment_batches = [[] for _ in range(num_gpus)]
        for i, segment in enumerate(segments):
            gpu_idx = i % num_gpus
            segment_batches[gpu_idx].append((segment, gpu_ids[gpu_idx]))

        outputs = [None] * len(segments)

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}

            for gpu_idx, batch in enumerate(segment_batches):
                for segment, gpu_id in batch:
                    future = executor.submit(
                        self._generate_on_gpu,
                        segment,
                        gpu_id
                    )
                    futures[future] = segment.segment_id

            # 收集结果
            for future in as_completed(futures):
                segment_id = futures[future]
                try:
                    output_path = future.result()
                    outputs[segment_id] = output_path
                    print(f"✓ Segment {segment_id} completed")
                except Exception as e:
                    print(f"✗ Segment {segment_id} failed: {e}")
                    raise

        return outputs

    def _generate_on_gpu(self, segment: SegmentConfig, gpu_id: int) -> Path:
        """在指定GPU上生成片段"""
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return self.generate_segment(segment)
```

---

### Phase 3: 视频拼接与后处理 (Week 3)

#### Step 3.1: VideoStitcher 实现
```python
import subprocess
from pathlib import Path
from typing import List

class VideoStitcher:
    def __init__(self, fps: int = 24, codec: str = "libx264", crf: int = 18):
        self.fps = fps
        self.codec = codec
        self.crf = crf

    def stitch_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        method: str = "concat",
        remove_intermediates: bool = False
    ) -> Path:
        """拼接视频片段"""

        if method == "concat":
            return self._concat_videos(video_paths, output_path, remove_intermediates)
        elif method == "blend":
            return self._blend_videos(video_paths, output_path)
        else:
            raise ValueError(f"Unknown stitching method: {method}")

    def _concat_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        remove_intermediates: bool = False
    ) -> Path:
        """使用 FFmpeg concat demuxer 拼接"""

        # 创建 concat 文件列表
        concat_list_file = output_path.parent / "concat_list.txt"
        with open(concat_list_file, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.resolve()}'\n")

        # 执行拼接（无损）
        output_raw = output_path.parent / (output_path.stem + "_raw.mp4")
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file),
            "-c", "copy",
            str(output_raw)
        ]

        print("Concatenating video segments...")
        subprocess.run(cmd_concat, check=True, capture_output=True)

        # 重新编码以确保兼容性
        cmd_encode = [
            "ffmpeg", "-y",
            "-i", str(output_raw),
            "-vf", f"fps={self.fps}",
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        print("Re-encoding for compatibility...")
        subprocess.run(cmd_encode, check=True, capture_output=True)

        # 清理中间文件
        if remove_intermediates:
            concat_list_file.unlink()
            output_raw.unlink()

        print(f"✓ Stitched video saved to: {output_path}")
        return output_path

    def _blend_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        blend_duration: float = 0.5
    ) -> Path:
        """使用交叉淡化拼接（实验性）"""

        # 构建复杂的 FFmpeg 滤镜链
        # 这里需要计算每个片段的时长，然后构建 xfade 滤镜

        # 示例：两个视频的交叉淡化
        if len(video_paths) == 2:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_paths[0]),
                "-i", str(video_paths[1]),
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={blend_duration}:offset=2.5[outv]",
                "-map", "[outv]",
                "-c:v", self.codec,
                "-crf", str(self.crf),
                str(output_path)
            ]
            subprocess.run(cmd, check=True)

        else:
            # 对于多个视频，需要递归构建滤镜链
            raise NotImplementedError("Blend mode for >2 videos not yet implemented")

        return output_path
```

#### Step 3.2: VideoSmoother 实现（基础版）
```python
class VideoSmoother:
    def __init__(self, method: str = "temporal_filter"):
        self.method = method

    def smooth_video(
        self,
        input_path: Path,
        output_path: Path,
        junction_frames: Optional[List[int]] = None
    ) -> Path:
        """平滑处理"""

        if self.method == "none":
            # 直接复制
            import shutil
            shutil.copy(input_path, output_path)
            return output_path

        elif self.method == "temporal_filter":
            return self._temporal_filter_smooth(input_path, output_path)

        elif self.method == "optical_flow":
            return self._optical_flow_smooth(input_path, output_path, junction_frames)

        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

    def _temporal_filter_smooth(
        self,
        input_path: Path,
        output_path: Path
    ) -> Path:
        """使用 FFmpeg 时域滤波器"""

        # 使用 minterpolate 或 tmix 滤镜
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", "tmix=frames=3:weights='1 2 1'",  # 简单时域平均
            "-c:v", "libx264",
            "-crf", "18",
            str(output_path)
        ]

        print("Applying temporal smoothing...")
        subprocess.run(cmd, check=True, capture_output=True)

        return output_path

    def _optical_flow_smooth(
        self,
        input_path: Path,
        output_path: Path,
        junction_frames: List[int]
    ) -> Path:
        """基于光流的局部平滑（拼接点附近）"""

        # 这需要更复杂的实现：
        # 1. 提取拼接点附近的帧
        # 2. 使用光流模型（如 RIFE）进行插帧
        # 3. 替换原始帧
        # 4. 重新编码视频

        print("Optical flow smoothing not yet fully implemented")
        print("Falling back to temporal filter...")

        return self._temporal_filter_smooth(input_path, output_path)
```

---

### Phase 4: 主流程编排与CLI (Week 4)

#### Step 4.1: Pipeline 实现
```python
# pipeline.py

from pathlib import Path
from typing import List, Optional, Tuple
from .keyframe_manager import KeyframeManager, GenerationConfig
from .segment_generator import FLF2VSegmentGenerator, ParallelFLF2VGenerator
from .video_stitcher import VideoStitcher
from .video_smoother import VideoSmoother

class MultiKeyframePipeline:
    """多关键帧视频生成主流程"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.keyframe_manager = None
        self.generator = None
        self.stitcher = VideoStitcher(fps=config.fps)
        self.smoother = None

    def run(
        self,
        keyframes: List[Tuple[str, float]],
        global_prompt: str,
        sub_prompts: Optional[List[str]] = None,
        output_path: str = "./output_final.mp4",
        num_gpus: int = 1,
        enable_smoothing: bool = True,
        smoothing_method: str = "temporal_filter"
    ) -> Path:
        """执行完整流程"""

        output_path = Path(output_path)

        # Step 1: 准备关键帧
        print("\n" + "="*60)
        print("STEP 1: Keyframe Preparation")
        print("="*60)

        self.keyframe_manager = KeyframeManager(keyframes, fps=self.config.fps)
        is_valid, message = self.keyframe_manager.validate()

        if not is_valid:
            raise ValueError(f"Keyframe validation failed: {message}")

        print(f"✓ Validation passed: {len(keyframes)} keyframes")

        # 获取片段配置
        segments = self.keyframe_manager.get_segments(global_prompt, sub_prompts)
        print(f"✓ Created {len(segments)} segments")

        # 检查帧数建议
        suggestions = self.keyframe_manager.suggest_frame_adjustments()
        for sug in suggestions:
            if sug["status"] != "ok":
                print(f"⚠ Segment {sug['segment_id']}: {sug['warning']}")

        # Step 2: 生成片段
        print("\n" + "="*60)
        print("STEP 2: Segment Generation")
        print("="*60)

        if num_gpus > 1:
            self.generator = ParallelFLF2VGenerator(self.config)
            video_paths = self.generator.generate_batch_parallel(segments, num_gpus)
        else:
            self.generator = FLF2VSegmentGenerator(self.config)
            video_paths = self.generator.generate_batch_sequential(segments)

        print(f"✓ All {len(video_paths)} segments generated")

        # Step 3: 拼接视频
        print("\n" + "="*60)
        print("STEP 3: Video Stitching")
        print("="*60)

        stitched_path = output_path.parent / (output_path.stem + "_stitched.mp4")
        self.stitcher.stitch_videos(video_paths, stitched_path)

        # Step 4: 后处理平滑
        if enable_smoothing:
            print("\n" + "="*60)
            print("STEP 4: Post-processing Smoothing")
            print("="*60)

            self.smoother = VideoSmoother(method=smoothing_method)

            # 计算拼接点帧号
            junction_frames = []
            cumulative_frames = 0
            for seg in segments[:-1]:
                cumulative_frames += seg.num_frames
                junction_frames.append(cumulative_frames)

            final_path = self.smoother.smooth_video(
                stitched_path,
                output_path,
                junction_frames
            )
        else:
            final_path = stitched_path
            final_path.rename(output_path)

        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED")
        print("="*60)
        print(f"Output video: {output_path}")

        return output_path
```

#### Step 4.2: 命令行工具
```python
# scripts/multi_keyframe_generate.py

import argparse
import yaml
from pathlib import Path
from wan.multi_keyframe import MultiKeyframePipeline, GenerationConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-keyframe video generation using Wan2.1 FLF2V"
    )

    # 输入配置
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (alternative to CLI args)"
    )
    parser.add_argument(
        "--keyframes",
        type=str,
        help="Keyframe list in format: 'path1:time1,path2:time2,...'"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Global prompt for all segments"
    )
    parser.add_argument(
        "--sub_prompts",
        type=str,
        help="Segment-specific prompts separated by '|'"
    )

    # 模型配置
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to FLF2V model checkpoint"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        help="Video size (width*height)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS"
    )

    # 生成参数
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1
    )

    # 并行与优化
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel generation"
    )
    parser.add_argument(
        "--offload_model",
        action="store_true",
        help="Offload model to CPU to save VRAM"
    )

    # 后处理
    parser.add_argument(
        "--enable_smoothing",
        action="store_true",
        default=True,
        help="Enable post-processing smoothing"
    )
    parser.add_argument(
        "--smoothing_method",
        type=str,
        default="temporal_filter",
        choices=["temporal_filter", "optical_flow", "none"]
    )

    # 输出
    parser.add_argument(
        "--output",
        type=str,
        default="./output_final.mp4",
        help="Output video path"
    )

    return parser.parse_args()

def load_config_from_yaml(config_path: str):
    """从 YAML 文件加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()

    # 如果提供了配置文件，则加载
    if args.config:
        config_dict = load_config_from_yaml(args.config)
        # 合并命令行参数（命令行优先）
        for key, value in vars(args).items():
            if value is not None or key not in config_dict:
                config_dict[key] = value
    else:
        config_dict = vars(args)

    # 解析关键帧列表
    if isinstance(config_dict['keyframes'], str):
        keyframes = []
        for item in config_dict['keyframes'].split(','):
            path, time = item.split(':')
            keyframes.append((path.strip(), float(time.strip())))
        config_dict['keyframes'] = keyframes

    # 解析子提示词
    sub_prompts = None
    if config_dict.get('sub_prompts'):
        sub_prompts = [p.strip() for p in config_dict['sub_prompts'].split('|')]

    # 创建生成配置
    gen_config = GenerationConfig(
        ckpt_dir=Path(config_dict['ckpt_dir']),
        size=config_dict['size'],
        fps=config_dict['fps'],
        sample_steps=config_dict.get('sample_steps'),
        sample_shift=config_dict.get('sample_shift'),
        sample_guide_scale=config_dict['sample_guide_scale'],
        base_seed=config_dict['base_seed'],
        offload_model=config_dict['offload_model']
    )

    # 运行流程
    pipeline = MultiKeyframePipeline(gen_config)

    output_path = pipeline.run(
        keyframes=config_dict['keyframes'],
        global_prompt=config_dict['prompt'],
        sub_prompts=sub_prompts,
        output_path=config_dict['output'],
        num_gpus=config_dict['num_gpus'],
        enable_smoothing=config_dict['enable_smoothing'],
        smoothing_method=config_dict['smoothing_method']
    )

    print(f"\n✓ Success! Video saved to: {output_path}")

if __name__ == "__main__":
    main()
```

---

## 四、改进与优化方向

### 4.1 质量优化

#### 优化 1: 自适应关键帧位置微调
**问题**：严格的首尾帧约束可能导致生成内容与关键帧不完全匹配

**方案**：
```python
class AdaptiveKeyframeAligner:
    """自适应关键帧对齐器"""

    def align_generated_frame(
        self,
        generated_frame: np.ndarray,
        target_keyframe: np.ndarray,
        blend_ratio: float = 0.8
    ) -> np.ndarray:
        """
        将生成的末帧与目标关键帧进行混合，
        确保下一个片段的首帧更加连贯
        """
        return (blend_ratio * generated_frame +
                (1 - blend_ratio) * target_keyframe)
```

#### 优化 2: 内容一致性增强
**问题**：长序列中角色、服装、场景可能不一致

**方案**：
- **提示词模板化**：使用固定模板强调一致性元素
  ```python
  CONSISTENCY_TEMPLATE = """
  Character: {character_description} (MUST remain consistent)
  Scene: {scene_description} (MUST remain consistent)
  Action: {action_description}
  Style: {style_description}
  """
  ```

- **参考图像引导**：在每个片段中添加角色/场景参考图
  ```python
  # 如果 FLF2V 支持额外的参考图输入
  segment_config.reference_images = [
      character_ref_image,
      scene_ref_image
  ]
  ```

#### 优化 3: 智能采样步数调整
**问题**：拼接点附近质量不稳定

**方案**：
```python
class AdaptiveSamplingScheduler:
    """自适应采样调度器"""

    def get_steps_for_segment(
        self,
        segment: SegmentConfig,
        base_steps: int = 40,
        boost_junction: bool = True
    ) -> int:
        """
        为拼接点附近的片段增加采样步数，
        提升关键帧附近的生成质量
        """
        if boost_junction:
            return int(base_steps * 1.25)  # 提升 25%
        return base_steps
```

---

### 4.2 效率优化

#### 优化 4: 智能缓存与复用
**问题**：重复生成浪费计算资源

**方案**：
```python
class SegmentCache:
    """片段缓存管理器"""

    def __init__(self, cache_dir: Path = Path("./segment_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, segment: SegmentConfig) -> str:
        """生成缓存键（基于内容哈希）"""
        import hashlib
        content = f"{segment.first_frame}{segment.last_frame}{segment.prompt}{segment.num_frames}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached(self, segment: SegmentConfig) -> Optional[Path]:
        """获取缓存的片段"""
        cache_key = self.get_cache_key(segment)
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        if cache_file.exists():
            print(f"✓ Cache hit for segment {segment.segment_id}")
            return cache_file
        return None

    def save_cache(self, segment: SegmentConfig, video_path: Path):
        """保存片段到缓存"""
        cache_key = self.get_cache_key(segment)
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        import shutil
        shutil.copy(video_path, cache_file)
```

#### 优化 5: 渐进式预览
**问题**：长时间等待无反馈

**方案**：
```python
class ProgressivePreviewGenerator:
    """渐进式预览生成器"""

    def generate_with_preview(
        self,
        segment: SegmentConfig,
        preview_steps: List[int] = [10, 20, 30, 40]
    ):
        """
        在生成过程中定期保存中间结果，
        允许用户提前查看效果
        """
        # 需要修改 generate.py 以支持中间步骤保存
        pass
```

---

### 4.3 高级功能

#### 功能 1: 动态时长分配
**问题**：用户提供的时间戳可能不适合视频生成

**方案**：
```python
class DynamicDurationAllocator:
    """动态时长分配器"""

    def optimize_durations(
        self,
        keyframes: List[KeyframeInfo],
        target_total_duration: float,
        min_segment_duration: float = 1.5,
        max_segment_duration: float = 4.0
    ) -> List[KeyframeInfo]:
        """
        根据内容复杂度自动调整片段时长：
        - 场景变化大的片段：分配更长时长
        - 场景变化小的片段：压缩时长
        """
        # 使用图像相似度计算场景变化程度
        from skimage.metrics import structural_similarity as ssim

        scene_changes = []
        for i in range(len(keyframes) - 1):
            img1 = self._load_image(keyframes[i].frame_path)
            img2 = self._load_image(keyframes[i+1].frame_path)
            similarity = ssim(img1, img2, channel_axis=-1)
            scene_changes.append(1 - similarity)  # 越不相似，变化越大

        # 归一化并分配时长
        total_change = sum(scene_changes)
        adjusted_keyframes = [keyframes[0]]

        cumulative_time = 0
        for i, change in enumerate(scene_changes):
            duration = (change / total_change) * target_total_duration
            duration = max(min_segment_duration, min(duration, max_segment_duration))

            cumulative_time += duration
            adjusted_keyframes.append(KeyframeInfo(
                frame_path=keyframes[i+1].frame_path,
                timestamp_sec=cumulative_time
            ))

        return adjusted_keyframes
```

#### 功能 2: 交互式关键帧编辑
**方案**：提供 GUI 工具进行可视化编辑
```python
# 使用 Gradio 构建简单界面
import gradio as gr

def create_keyframe_editor():
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Keyframe Video Generator")

        with gr.Row():
            with gr.Column():
                keyframe_gallery = gr.Gallery(label="Keyframes")
                upload_btn = gr.UploadButton("Add Keyframe", file_count="multiple")

                timeline = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=0.1,
                    label="Timeline (seconds)"
                )

            with gr.Column():
                global_prompt = gr.Textbox(label="Global Prompt")
                segment_prompts = gr.Textbox(
                    label="Segment Prompts (one per line)",
                    lines=5
                )

                generate_btn = gr.Button("Generate Video")
                output_video = gr.Video(label="Output")

        # 绑定事件...

    return demo
```

#### 功能 3: 风格迁移与混合
**方案**：在不同片段间进行风格渐变
```python
class StyleTransitionGenerator:
    """风格过渡生成器"""

    def generate_with_style_transition(
        self,
        segments: List[SegmentConfig],
        style_prompts: List[str]  # 每段的风格描述
    ):
        """
        在相邻片段间插入风格过渡片段，
        例如：写实风格 → 卡通风格 的平滑转换
        """
        # 在拼接点插入混合风格的短片段
        # prompt = f"0.7 * style_1 + 0.3 * style_2"
        pass
```

---

## 五、测试与验证

### 5.1 单元测试
```python
# tests/test_keyframe_manager.py

import unittest
from wan.multi_keyframe import KeyframeManager

class TestKeyframeManager(unittest.TestCase):

    def test_validation_file_not_found(self):
        keyframes = [
            ("nonexistent.png", 0.0),
            ("also_nonexistent.png", 2.0)
        ]
        manager = KeyframeManager(keyframes)
        is_valid, message = manager.validate()
        self.assertFalse(is_valid)
        self.assertIn("not found", message.lower())

    def test_validation_wrong_order(self):
        # 假设这些文件存在
        keyframes = [
            ("frame2.png", 2.0),
            ("frame1.png", 1.0)  # 时间顺序错误
        ]
        manager = KeyframeManager(keyframes)
        is_valid, message = manager.validate()
        self.assertFalse(is_valid)
        self.assertIn("chronological", message.lower())

    def test_segment_generation(self):
        keyframes = [
            ("frame1.png", 0.0),
            ("frame2.png", 2.0),
            ("frame3.png", 4.0)
        ]
        manager = KeyframeManager(keyframes, fps=24)
        segments = manager.get_segments("global prompt")

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].num_frames, 49)  # 2s * 24fps + 1
        self.assertEqual(segments[1].num_frames, 49)
```

### 5.2 集成测试
```python
# tests/test_pipeline.py

import unittest
from pathlib import Path
from wan.multi_keyframe import MultiKeyframePipeline, GenerationConfig

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.config = GenerationConfig(
            ckpt_dir=Path("./test_ckpt"),
            size="640*480",  # 使用小尺寸加速测试
            fps=12,
            sample_steps=10  # 减少步数加速测试
        )

    def test_full_pipeline(self):
        keyframes = [
            ("tests/fixtures/frame1.png", 0.0),
            ("tests/fixtures/frame2.png", 1.0)
        ]

        pipeline = MultiKeyframePipeline(self.config)
        output = pipeline.run(
            keyframes=keyframes,
            global_prompt="test prompt",
            output_path="./test_output.mp4",
            num_gpus=1,
            enable_smoothing=False
        )

        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)
```

---

## 六、使用示例

### 示例 1: 基础命令行使用
```bash
python scripts/multi_keyframe_generate.py \
  --keyframes "scene1.png:0.0,scene2.png:3.0,scene3.png:6.0,scene4.png:10.0" \
  --prompt "cinematic style, soft lighting, natural motion, 4k" \
  --ckpt_dir ./Wan2.1-FLF2V-14B-720P \
  --size 1280*720 \
  --fps 24 \
  --output ./my_video.mp4 \
  --enable_smoothing
```

### 示例 2: YAML 配置文件
```yaml
# config.yaml

keyframes:
  - ["keyframes/frame_001.png", 0.0]
  - ["keyframes/frame_002.png", 2.5]
  - ["keyframes/frame_003.png", 5.0]
  - ["keyframes/frame_004.png", 8.0]

prompt: "A beautiful journey through magical landscapes, cinematic style, 4k quality"

sub_prompts:
  - "Starting in a misty forest at dawn"
  - "Transitioning to a sunlit meadow with butterflies"
  - "Ending at a majestic mountain peak at sunset"

ckpt_dir: "./Wan2.1-FLF2V-14B-720P"
size: "1280*720"
fps: 24

sample_steps: 50
sample_shift: 16.0
sample_guide_scale: 5.0
base_seed: 42

num_gpus: 4
offload_model: false

enable_smoothing: true
smoothing_method: "temporal_filter"

output: "./outputs/magical_journey.mp4"
```

使用配置文件：
```bash
python scripts/multi_keyframe_generate.py --config config.yaml
```

### 示例 3: Python API 使用
```python
from pathlib import Path
from wan.multi_keyframe import MultiKeyframePipeline, GenerationConfig

# 配置
config = GenerationConfig(
    ckpt_dir=Path("./Wan2.1-FLF2V-14B-720P"),
    size="1280*720",
    fps=24,
    sample_steps=50,
    sample_shift=16.0,
    sample_guide_scale=5.0,
    base_seed=42
)

# 定义关键帧
keyframes = [
    ("keyframes/sunrise.png", 0.0),
    ("keyframes/noon.png", 5.0),
    ("keyframes/sunset.png", 10.0)
]

# 创建流程
pipeline = MultiKeyframePipeline(config)

# 运行
output = pipeline.run(
    keyframes=keyframes,
    global_prompt="A day in the life of a peaceful village, cinematic, 4k",
    sub_prompts=[
        "Morning: villagers waking up, roosters crowing",
        "Afternoon: bustling market, children playing"
    ],
    output_path="./village_day.mp4",
    num_gpus=2,
    enable_smoothing=True
)

print(f"Video generated: {output}")
```

---

## 七、常见问题与解决方案

### Q1: 片段拼接处出现明显闪烁
**原因**：
- 相邻片段的末帧和首帧不完全一致
- 生成的动态在拼接点突变

**解决方案**：
1. **启用平滑处理**：
   ```bash
   --enable_smoothing --smoothing_method temporal_filter
   ```

2. **增加采样步数**（提升关键帧附近质量）：
   ```bash
   --sample_steps 60  # 默认 40-50
   ```

3. **使用交叉淡化拼接**：
   ```python
   stitcher.stitch_videos(videos, output, method="blend", blend_duration=0.3)
   ```

### Q2: 内容一致性差（角色/场景变化）
**原因**：
- 提示词不够具体
- 片段过长导致概念漂移

**解决方案**：
1. **强化提示词一致性元素**：
   ```python
   prompt = """
   Character: {固定的角色描述，包括外貌、服装}
   Scene: {固定的场景描述}
   Action: {当前片段的动作}
   Style: cinematic, 4k, consistent lighting
   """
   ```

2. **控制片段长度**：
   ```python
   # 建议每段 32-80 帧 (1.3-3.3秒 @ 24fps)
   max_frames_per_segment = 80
   ```

### Q3: 生成速度太慢
**解决方案**：
1. **使用多GPU并行**：
   ```bash
   --num_gpus 4
   ```

2. **降低分辨率或帧数**：
   ```bash
   --size 960*540  # 从 1280*720 降低
   --frame_num 49  # 减少每段帧数
   ```

3. **启用模型卸载（牺牲速度换显存）**：
   ```bash
   --offload_model
   ```

### Q4: 显存不足 (OOM)
**解决方案**：
1. **使用 CPU 卸载**：
   ```bash
   --offload_model --t5_cpu
   ```

2. **减小批次大小**（顺序生成而非并行）：
   ```bash
   --num_gpus 1
   ```

3. **使用更小的模型**（如果可用）：
   ```bash
   # 使用 1.3B 而非 14B
   --task flf2v-1.3B --ckpt_dir ./Wan2.1-FLF2V-1.3B
   ```

---

## 八、未来路线图

### 短期 (1-2 个月)
- [ ] 实现基础四模块功能
- [ ] 完成命令行工具和 Python API
- [ ] 集成基础平滑处理
- [ ] 编写单元测试和文档

### 中期 (3-6 个月)
- [ ] 支持 RIFE 光流插帧平滑
- [ ] 实现自适应采样调度
- [ ] 添加片段缓存机制
- [ ] 开发 Gradio 可视化界面
- [ ] 支持动态时长分配

### 长期 (6+ 个月)
- [ ] 集成风格迁移功能
- [ ] 支持音频同步生成
- [ ] 开发高级一致性控制（角色/场景锁定）
- [ ] 支持实时预览和交互式编辑
- [ ] 优化为生产级工具（错误恢复、日志、监控）

---

## 九、参考资料

### 相关论文
- Flow Matching for Generative Modeling
- Ulysses: Efficient Sequence Parallel Attention
- RIFE: Real-Time Intermediate Flow Estimation

### 工具与库
- **FFmpeg**: 视频处理和拼接
- **RIFE**: 光流插帧 (https://github.com/megvii-research/ECCV2022-RIFE)
- **xDiT**: 序列并行加速 (https://github.com/xdit-project/xDiT)
- **Gradio**: 快速构建 UI (https://gradio.app)

### Wan2.1 相关
- 官方仓库: https://github.com/Wan-Video/Wan2.1
- 技术报告: https://arxiv.org/abs/2503.20314
- 模型下载: https://huggingface.co/Wan-AI

---

## 十、总结

本实现方案提供了一个**完整、可扩展、易于实施**的多关键帧视频合成工作流。核心优势包括：

1. **模块化设计**：各模块职责清晰，便于测试和维护
2. **渐进式实现**：从基础功能到高级优化，分阶段推进
3. **灵活配置**：支持命令行、配置文件、Python API 多种使用方式
4. **性能优化**：原生支持多GPU并行，可扩展到大规模生成
5. **质量保障**：包含多种平滑和一致性增强策略

建议按照 Phase 1-4 的顺序逐步实现，每个阶段完成后进行充分测试，再进入下一阶段。
