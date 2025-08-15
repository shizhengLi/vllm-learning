# vLLM教育版简化实现方案：理解LLM推理的核心原理

## 1. 概述

本文档提供了一个简化版的vLLM实现方案，专门用于教育和学习目的。这个版本保留了vLLM的核心思想和关键特性，但大大简化了实现复杂度，使学习者能够更容易理解和掌握LLM推理的核心原理。

### 1.1 教育目标

**核心概念理解**:
- 理解PagedAttention的核心思想
- 掌握连续批处理的调度原理
- 学习LLM推理的基本流程
- 了解性能优化的基本策略

**实践技能培养**:
- 能够实现基本的LLM推理引擎
- 理解内存管理和调度算法
- 掌握性能分析和优化方法

### 1.2 技术栈选择

**核心依赖**:
- **深度学习**: PyTorch (简化版，不需要CUDA)
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib (可选)
- **测试**: pytest

**开发环境**:
- Python 3.8+
- PyTorch 1.12+
- 标准库（不依赖复杂的第三方库）

## 2. 系统架构设计

### 2.1 简化架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户接口层                               │
├─────────────────────────────────────────────────────────────┤
│                 SimpleLLM (简化LLM类)                          │
├─────────────────────────────────────────────────────────────┤
│                      简化引擎层                               │
├─────────────────────────────────────────────────────────────┤
│  SimpleEngine  │  SimpleScheduler  │  SimpleMemoryManager      │
├─────────────────────────────────────────────────────────────┤
│                      简化执行层                               │
├─────────────────────────────────────────────────────────────┤
│  SimpleModelRunner  │  SimpleAttention  │  SimpleCache          │
├─────────────────────────────────────────────────────────────┤
│                      基础层                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **简化优先**: 移除复杂的优化和分布式功能
2. **教育导向**: 代码结构清晰，注释详细
3. **逐步演进**: 从简单到复杂，便于理解
4. **可视化支持**: 提供性能和内存使用可视化

## 3. 核心组件实现

### 3.1 简化的配置管理

```python
# simple_config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class SimpleModelConfig:
    """简化的模型配置"""
    model_name: str = "facebook/opt-125m"
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    max_length: int = 512
    
    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead")
            self.device = "cpu"

@dataclass
class SimpleCacheConfig:
    """简化的缓存配置"""
    block_size: int = 16
    max_blocks: int = 100
    enable_caching: bool = True
    
    def __post_init__(self):
        if self.block_size <= 0:
            raise ValueError("Block size must be positive")
        if self.max_blocks <= 0:
            raise ValueError("Max blocks must be positive")

@dataclass
class SimpleSchedulerConfig:
    """简化的调度器配置"""
    max_batch_size: int = 8
    max_tokens: int = 1024
    enable_chunking: bool = True
    chunk_size: int = 64

@dataclass
class SimpleVllmConfig:
    """统一的简化配置"""
    model_config: SimpleModelConfig = field(default_factory=SimpleModelConfig)
    cache_config: SimpleCacheConfig = field(default_factory=SimpleCacheConfig)
    scheduler_config: SimpleSchedulerConfig = field(default_factory=SimpleSchedulerConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SimpleVllmConfig":
        """从字典创建配置"""
        model_config = SimpleModelConfig(**config_dict.get("model", {}))
        cache_config = SimpleCacheConfig(**config_dict.get("cache", {}))
        scheduler_config = SimpleSchedulerConfig(**config_dict.get("scheduler", {}))
        
        return cls(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config
        )
```

### 3.2 简化的序列管理

```python
# simple_sequence.py
from dataclasses import dataclass, field
from typing import List, Optional
import time
from enum import Enum

class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"

@dataclass
class SimpleSequence:
    """简化的序列类"""
    
    # 基础信息
    seq_id: str
    prompt: str
    token_ids: List[int]
    
    # 状态管理
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    
    # 生成结果
    generated_token_ids: List[int] = field(default_factory=list)
    generated_text: str = ""
    
    # 采样参数
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 100
    
    def __post_init__(self):
        if not self.token_ids:
            raise ValueError("Token IDs cannot be empty")
    
    def get_length(self) -> int:
        """获取序列长度"""
        return len(self.token_ids) + len(self.generated_token_ids)
    
    def get_num_generated_tokens(self) -> int:
        """获取已生成的token数量"""
        return len(self.generated_token_ids)
    
    def is_finished(self) -> bool:
        """检查序列是否完成"""
        return self.status in [SequenceStatus.FINISHED, SequenceStatus.ABORTED]
    
    def is_running(self) -> bool:
        """检查序列是否正在运行"""
        return self.status == SequenceStatus.RUNNING
    
    def start_generation(self):
        """开始生成"""
        self.status = SequenceStatus.RUNNING
        self.start_time = time.time()
    
    def finish_generation(self):
        """完成生成"""
        self.status = SequenceStatus.FINISHED
        self.finish_time = time.time()
    
    def abort_generation(self):
        """中断生成"""
        self.status = SequenceStatus.ABORTED
        self.finish_time = time.time()
    
    def add_generated_token(self, token_id: int, token_text: str):
        """添加生成的token"""
        self.generated_token_ids.append(token_id)
        self.generated_text += token_text
    
    def get_generation_time(self) -> float:
        """获取生成时间"""
        if self.start_time is None:
            return 0.0
        end_time = self.finish_time or time.time()
        return end_time - self.start_time
    
    def get_tokens_per_second(self) -> float:
        """获取每秒token数"""
        gen_time = self.get_generation_time()
        if gen_time == 0:
            return 0.0
        return self.get_num_generated_tokens() / gen_time
    
    def to_dict(self) -> dict:
        """转换为字典（用于调试和可视化）"""
        return {
            "seq_id": self.seq_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "arrival_time": self.arrival_time,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "input_tokens": len(self.token_ids),
            "generated_tokens": len(self.generated_token_ids),
            "generated_text": self.generated_text,
            "generation_time": self.get_generation_time(),
            "tokens_per_second": self.get_tokens_per_second()
        }

class SimpleSequenceGroup:
    """简化的序列组（用于批处理）"""
    
    def __init__(self, sequences: List[SimpleSequence]):
        self.sequences = sequences
        self.group_id = f"group_{int(time.time() * 1000)}"
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, index: int) -> SimpleSequence:
        return self.sequences[index]
    
    def get_total_tokens(self) -> int:
        """获取总token数"""
        return sum(seq.get_length() for seq in self.sequences)
    
    def get_num_generated_tokens(self) -> int:
        """获取总生成token数"""
        return sum(seq.get_num_generated_tokens() for seq in self.sequences)
    
    def get_running_sequences(self) -> List[SimpleSequence]:
        """获取正在运行的序列"""
        return [seq for seq in self.sequences if seq.is_running()]
    
    def get_finished_sequences(self) -> List[SimpleSequence]:
        """获取已完成的序列"""
        return [seq for seq in self.sequences if seq.is_finished()]
    
    def start_generation(self):
        """开始所有序列的生成"""
        for seq in self.sequences:
            if not seq.is_finished():
                seq.start_generation()
    
    def add_generated_tokens(self, token_ids: List[List[int]], token_texts: List[str]):
        """添加生成的tokens"""
        for i, seq in enumerate(self.sequences):
            if seq.is_running() and i < len(token_ids):
                token_id = token_ids[i][0] if token_ids[i] else 0
                token_text = token_texts[i] if i < len(token_texts) else ""
                seq.add_generated_token(token_id, token_text)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "group_id": self.group_id,
            "num_sequences": len(self.sequences),
            "total_tokens": self.get_total_tokens(),
            "generated_tokens": self.get_num_generated_tokens(),
            "sequences": [seq.to_dict() for seq in self.sequences]
        }
```

### 3.3 简化的PagedAttention实现

```python
# simple_paged_attention.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SimpleBlock:
    """简化的内存块"""
    block_id: int
    token_ids: List[int]
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None
    ref_count: int = 1
    
    def __post_init__(self):
        if not self.token_ids:
            raise ValueError("Token IDs cannot be empty")
    
    def get_num_tokens(self) -> int:
        """获取块中的token数量"""
        return len(self.token_ids)
    
    def is_full(self, max_size: int) -> bool:
        """检查块是否已满"""
        return len(self.token_ids) >= max_size
    
    def add_token(self, token_id: int, key: torch.Tensor, value: torch.Tensor):
        """添加token到块中"""
        if self.is_full(len(self.token_ids) + 1):
            raise ValueError("Block is full")
        
        self.token_ids.append(token_id)
        
        # 初始化缓存
        if self.key_cache is None:
            self.key_cache = torch.zeros_like(key)
            self.value_cache = torch.zeros_like(value)
        
        # 添加到缓存
        token_pos = len(self.token_ids) - 1
        self.key_cache[token_pos] = key
        self.value_cache[token_pos] = value
    
    def get_key_value(self, token_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定位置的key和value"""
        if token_pos >= len(self.token_ids):
            raise ValueError("Token position out of range")
        
        return self.key_cache[token_pos], self.value_cache[token_pos]
    
    def increase_ref_count(self):
        """增加引用计数"""
        self.ref_count += 1
    
    def decrease_ref_count(self):
        """减少引用计数"""
        self.ref_count -= 1
        return self.ref_count

class SimplePagedAttention:
    """简化的PagedAttention实现"""
    
    def __init__(self, block_size: int = 16, max_blocks: int = 100):
        self.block_size = block_size
        self.max_blocks = max_blocks
        
        # 块管理
        self.blocks: Dict[int, SimpleBlock] = {}
        self.free_blocks: List[int] = list(range(max_blocks))
        self.next_block_id = 0
        
        # 序列到块的映射
        self.seq_block_mapping: Dict[str, List[int]] = {}
        
        # 统计信息
        self.stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fragmentation_ratio": 0.0
        }
    
    def allocate_blocks(self, seq_id: str, num_tokens: int) -> List[int]:
        """为序列分配块"""
        
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        # 检查是否有足够的空闲块
        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError(f"Insufficient blocks: needed {num_blocks_needed}, available {len(self.free_blocks)}")
        
        # 分配块
        allocated_blocks = []
        for i in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated_blocks.append(block_id)
            
            # 创建新块
            self.blocks[block_id] = SimpleBlock(
                block_id=block_id,
                token_ids=[]
            )
        
        # 记录映射
        self.seq_block_mapping[seq_id] = allocated_blocks
        
        # 更新统计
        self.stats["total_allocations"] += num_blocks_needed
        
        return allocated_blocks
    
    def free_blocks(self, seq_id: str):
        """释放序列占用的块"""
        
        if seq_id not in self.seq_block_mapping:
            return
        
        block_ids = self.seq_block_mapping[seq_id]
        del self.seq_block_mapping[seq_id]
        
        for block_id in block_ids:
            if block_id in self.blocks:
                block = self.blocks[block_id]
                block.decrease_ref_count()
                
                # 如果引用计数为0，则真正释放
                if block.ref_count <= 0:
                    del self.blocks[block_id]
                    self.free_blocks.append(block_id)
                    self.stats["total_frees"] += 1
        
        # 更新碎片化统计
        self._update_fragmentation_stats()
    
    def store_kv_cache(self, seq_id: str, token_ids: List[int], 
                       keys: torch.Tensor, values: torch.Tensor):
        """存储KV缓存"""
        
        if seq_id not in self.seq_block_mapping:
            # 首次分配块
            self.allocate_blocks(seq_id, len(token_ids))
        
        block_ids = self.seq_block_mapping[seq_id]
        
        # 存储到块中
        for i, token_id in enumerate(token_ids):
            block_idx = i // self.block_size
            block_pos = i % self.block_size
            
            if block_idx < len(block_ids):
                block_id = block_ids[block_idx]
                block = self.blocks[block_id]
                block.add_token(token_id, keys[i], values[i])
    
    def retrieve_kv_cache(self, seq_id: str, start_pos: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索KV缓存"""
        
        if seq_id not in self.seq_block_mapping:
            # 返回空张量
            empty_shape = (0, keys.shape[-1] if 'keys' in locals() else 64)
            return torch.empty(empty_shape), torch.empty(empty_shape)
        
        block_ids = self.seq_block_mapping[seq_id]
        all_keys = []
        all_values = []
        
        for token_pos in range(start_pos, end_pos):
            block_idx = token_pos // self.block_size
            block_offset = token_pos % self.block_size
            
            if block_idx < len(block_ids):
                block_id = block_ids[block_idx]
                block = self.blocks[block_id]
                
                if block_offset < len(block.token_ids):
                    key, value = block.get_key_value(block_offset)
                    all_keys.append(key)
                    all_values.append(value)
        
        if all_keys:
            return torch.stack(all_keys), torch.stack(all_values)
        else:
            empty_shape = (0, 64)  # 假设head_size为64
            return torch.empty(empty_shape), torch.empty(empty_shape)
    
    def compute_attention(self, query: torch.Tensor, keys: torch.Tensor, 
                          values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """计算注意力（简化版）"""
        
        if keys.numel() == 0:
            return torch.zeros_like(query)
        
        # QK^T
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Attention输出
        output = torch.matmul(attn_probs, values)
        
        return output
    
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        total_blocks = len(self.blocks)
        free_blocks = len(self.free_blocks)
        used_blocks = total_blocks - free_blocks
        
        return {
            "total_blocks": self.max_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "utilization": used_blocks / self.max_blocks if self.max_blocks > 0 else 0
        }
    
    def get_stats(self) -> Dict[str, any]:
        """获取统计信息"""
        memory_usage = self.get_memory_usage()
        
        return {
            **self.stats,
            **memory_usage,
            "efficiency": self._calculate_efficiency()
        }
    
    def _update_fragmentation_stats(self):
        """更新碎片化统计"""
        if not self.free_blocks:
            self.stats["fragmentation_ratio"] = 0.0
            return
        
        # 计算碎片化程度
        free_contiguous = 0
        max_contiguous = 0
        current_contiguous = 0
        
        for i in range(self.max_blocks):
            if i in self.free_blocks:
                current_contiguous += 1
                max_contiguous = max(max_contiguous, current_contiguous)
            else:
                current_contiguous = 0
        
        if len(self.free_blocks) > 0:
            self.stats["fragmentation_ratio"] = 1.0 - (max_contiguous / len(self.free_blocks))
        else:
            self.stats["fragmentation_ratio"] = 0.0
    
    def _calculate_efficiency(self) -> float:
        """计算内存效率"""
        if self.stats["total_allocations"] == 0:
            return 0.0
        
        # 简单的效率计算
        utilization = self.get_memory_usage()["utilization"]
        fragmentation = 1.0 - self.stats["fragmentation_ratio"]
        
        return utilization * fragmentation
    
    def visualize_memory_layout(self):
        """可视化内存布局（教育用途）"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 创建内存映射
            memory_map = np.zeros(self.max_blocks)
            
            # 标记已使用的块
            for block_id, block in self.blocks.items():
                if block_id < self.max_blocks:
                    memory_map[block_id] = 1
            
            # 标记空闲块
            for block_id in self.free_blocks:
                if block_id < self.max_blocks:
                    memory_map[block_id] = 0
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 内存使用图
            ax1.imshow(memory_map.reshape(1, -1), cmap='RdYlGn', aspect='auto')
            ax1.set_title('Memory Layout (Green=Free, Red=Used)')
            ax1.set_xlabel('Block ID')
            ax1.set_yticks([])
            
            # 添加块ID标签
            for i in range(self.max_blocks):
                ax1.text(i, 0, str(i), ha='center', va='center', fontsize=8)
            
            # 统计信息图
            stats = self.get_stats()
            categories = ['Used', 'Free', 'Fragmented']
            values = [
                stats['used_blocks'],
                stats['free_blocks'],
                stats['fragmentation_ratio'] * stats['free_blocks']
            ]
            
            ax2.bar(categories, values, color=['red', 'green', 'orange'])
            ax2.set_title('Memory Statistics')
            ax2.set_ylabel('Number of Blocks')
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
    
    def print_debug_info(self):
        """打印调试信息"""
        print("=" * 50)
        print("SimplePagedAttention Debug Info")
        print("=" * 50)
        
        stats = self.get_stats()
        print(f"Total Blocks: {stats['total_blocks']}")
        print(f"Used Blocks: {stats['used_blocks']}")
        print(f"Free Blocks: {stats['free_blocks']}")
        print(f"Utilization: {stats['utilization']:.2%}")
        print(f"Fragmentation: {stats['fragmentation_ratio']:.2%}")
        print(f"Efficiency: {stats['efficiency']:.2%}")
        print(f"Total Allocations: {stats['total_allocations']}")
        print(f"Total Frees: {stats['total_frees']}")
        
        print("\nBlock Details:")
        for block_id, block in self.blocks.items():
            print(f"  Block {block_id}: {len(block.token_ids)} tokens, ref_count={block.ref_count}")
        
        print("\nSequence Mapping:")
        for seq_id, block_ids in self.seq_block_mapping.items():
            print(f"  {seq_id}: {block_ids}")
        
        print("=" * 50)
```

### 3.4 简化的调度器

```python
# simple_scheduler.py
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import heapq

from simple_sequence import SimpleSequence, SimpleSequenceGroup, SequenceStatus
from simple_paged_attention import SimplePagedAttention

@dataclass
class SimpleSchedulingBudget:
    """简化的调度预算"""
    max_batch_size: int
    max_tokens: int
    current_batch_size: int = 0
    current_tokens: int = 0
    
    def can_schedule(self, num_sequences: int, num_tokens: int) -> bool:
        """检查是否可以调度"""
        return (self.current_batch_size + num_sequences <= self.max_batch_size and
                self.current_tokens + num_tokens <= self.max_tokens)
    
    def schedule(self, num_sequences: int, num_tokens: int):
        """调度资源"""
        self.current_batch_size += num_sequences
        self.current_tokens += num_tokens
    
    def free(self, num_sequences: int, num_tokens: int):
        """释放资源"""
        self.current_batch_size = max(0, self.current_batch_size - num_sequences)
        self.current_tokens = max(0, self.current_tokens - num_tokens)

@dataclass
class SimpleSchedulerOutput:
    """简化的调度器输出"""
    scheduled_groups: List[SimpleSequenceGroup]
    ignored_groups: List[SimpleSequenceGroup]
    budget_utilization: float
    
class SimpleScheduler:
    """简化的调度器"""
    
    def __init__(self, max_batch_size: int = 8, max_tokens: int = 1024, 
                 enable_chunking: bool = True, chunk_size: int = 64):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size
        
        # 队列管理
        self.waiting_queue = deque()
        self.running_queue = deque()
        
        # 内存管理
        self.memory_manager = SimplePagedAttention()
        
        # 统计信息
        self.stats = {
            "total_scheduled": 0,
            "total_completed": 0,
            "total_chunks": 0,
            "avg_batch_size": 0.0,
            "avg_tokens_per_batch": 0.0,
            "scheduling_efficiency": 0.0
        }
        
        # 调度历史（用于可视化）
        self.scheduling_history = []
    
    def add_sequence(self, sequence: SimpleSequence):
        """添加序列到等待队列"""
        self.waiting_queue.append(sequence)
    
    def add_sequence_group(self, group: SimpleSequenceGroup):
        """添加序列组到等待队列"""
        self.waiting_queue.append(group)
    
    def schedule_step(self) -> SimpleSchedulerOutput:
        """执行一个调度步骤"""
        
        # 1. 创建调度预算
        budget = SimpleSchedulingBudget(
            max_batch_size=self.max_batch_size,
            max_tokens=self.max_tokens,
            current_batch_size=len(self.running_queue),
            current_tokens=self._get_current_tokens()
        )
        
        # 2. 处理完成的序列
        self._process_completed_sequences()
        
        # 3. 调度新序列
        scheduled_groups = self._schedule_new_sequences(budget)
        
        # 4. 处理内存压力
        self._handle_memory_pressure()
        
        # 5. 创建调度输出
        output = SimpleSchedulerOutput(
            scheduled_groups=scheduled_groups,
            ignored_groups=list(self.waiting_queue),
            budget_utilization=self._calculate_budget_utilization(budget)
        )
        
        # 6. 更新统计
        self._update_stats(scheduled_groups, budget)
        
        # 7. 记录历史
        self._record_scheduling_history(output)
        
        return output
    
    def _schedule_new_sequences(self, budget: SimpleSchedulingBudget) -> List[SimpleSequenceGroup]:
        """调度新序列"""
        scheduled_groups = []
        
        # 复制等待队列以便操作
        waiting_copy = list(self.waiting_queue)
        
        for item in waiting_copy:
            if not self._can_schedule_item(item, budget):
                continue
            
            # 检查是否需要分块
            if self._needs_chunking(item):
                chunked_groups = self._create_chunked_groups(item)
                
                # 调度分块
                for chunk_group in chunked_groups:
                    if self._can_schedule_item(chunk_group, budget):
                        self._schedule_item(chunk_group, budget)
                        scheduled_groups.append(chunk_group)
                        self.waiting_queue.remove(item)
                        break
            else:
                # 直接调度
                self._schedule_item(item, budget)
                scheduled_groups.append(item)
                self.waiting_queue.remove(item)
        
        return scheduled_groups
    
    def _can_schedule_item(self, item, budget: SimpleSchedulingBudget) -> bool:
        """检查是否可以调度项目"""
        if isinstance(item, SimpleSequence):
            num_sequences = 1
            num_tokens = item.get_length()
        else:  # SimpleSequenceGroup
            num_sequences = len(item)
            num_tokens = item.get_total_tokens()
        
        return budget.can_schedule(num_sequences, num_tokens)
    
    def _needs_chunking(self, item) -> bool:
        """检查是否需要分块"""
        if not self.enable_chunking:
            return False
        
        if isinstance(item, SimpleSequence):
            num_tokens = item.get_length()
        else:  # SimpleSequenceGroup
            num_tokens = item.get_total_tokens()
        
        return num_tokens > self.chunk_size
    
    def _create_chunked_groups(self, item) -> List[SimpleSequenceGroup]:
        """创建分块组"""
        if isinstance(item, SimpleSequence):
            sequences = [item]
        else:  # SimpleSequenceGroup
            sequences = item.sequences
        
        chunked_groups = []
        
        for seq in sequences:
            # 计算需要的块数
            total_tokens = seq.get_length()
            num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
            
            for chunk_idx in range(num_chunks):
                start_token = chunk_idx * self.chunk_size
                end_token = min(start_token + self.chunk_size, total_tokens)
                
                # 创建分块序列
                chunked_seq = SimpleSequence(
                    seq_id=f"{seq.seq_id}_chunk_{chunk_idx}",
                    prompt=seq.prompt,
                    token_ids=seq.token_ids[start_token:end_token],
                    temperature=seq.temperature,
                    top_p=seq.top_p,
                    max_tokens=seq.max_tokens
                )
                
                chunked_group = SimpleSequenceGroup([chunked_seq])
                chunked_groups.append(chunked_group)
        
        self.stats["total_chunks"] += len(chunked_groups)
        return chunked_groups
    
    def _schedule_item(self, item, budget: SimpleSchedulingBudget):
        """调度项目"""
        if isinstance(item, SimpleSequence):
            num_sequences = 1
            num_tokens = item.get_length()
            
            # 分配内存
            try:
                self.memory_manager.store_kv_cache(
                    item.seq_id, 
                    item.token_ids,
                    torch.zeros(len(item.token_ids), 64),  # 简化的key
                    torch.zeros(len(item.token_ids), 64)   # 简化的value
                )
                item.start_generation()
                self.running_queue.append(item)
                budget.schedule(num_sequences, num_tokens)
            except MemoryError:
                # 内存不足，跳过
                pass
                
        else:  # SimpleSequenceGroup
            num_sequences = len(item)
            num_tokens = item.get_total_tokens()
            
            # 为组中的每个序列分配内存
            try:
                for seq in item.sequences:
                    self.memory_manager.store_kv_cache(
                        seq.seq_id,
                        seq.token_ids,
                        torch.zeros(len(seq.token_ids), 64),
                        torch.zeros(len(seq.token_ids), 64)
                    )
                
                item.start_generation()
                self.running_queue.append(item)
                budget.schedule(num_sequences, num_tokens)
            except MemoryError:
                # 内存不足，释放已分配的内存
                for seq in item.sequences:
                    self.memory_manager.free_blocks(seq.seq_id)
                pass
    
    def _process_completed_sequences(self):
        """处理完成的序列"""
        completed_sequences = []
        
        for item in list(self.running_queue):
            if isinstance(item, SimpleSequence):
                if item.is_finished():
                    completed_sequences.append(item)
            else:  # SimpleSequenceGroup
                finished_seqs = item.get_finished_sequences()
                if len(finished_seqs) == len(item):
                    completed_sequences.append(item)
        
        # 移除完成的序列
        for item in completed_sequences:
            self.running_queue.remove(item)
            
            # 释放内存
            if isinstance(item, SimpleSequence):
                self.memory_manager.free_blocks(item.seq_id)
                self.stats["total_completed"] += 1
            else:  # SimpleSequenceGroup
                for seq in item.sequences:
                    self.memory_manager.free_blocks(seq.seq_id)
                self.stats["total_completed"] += len(item)
    
    def _handle_memory_pressure(self):
        """处理内存压力"""
        # 简化的内存压力处理
        memory_usage = self.memory_manager.get_memory_usage()
        
        if memory_usage["utilization"] > 0.9:  # 90%使用率阈值
            # 尝试释放一些内存
            self._cleanup_unused_memory()
    
    def _cleanup_unused_memory(self):
        """清理未使用的内存"""
        # 简化的清理逻辑
        pass
    
    def _get_current_tokens(self) -> int:
        """获取当前token数量"""
        total_tokens = 0
        
        for item in self.running_queue:
            if isinstance(item, SimpleSequence):
                total_tokens += item.get_length()
            else:  # SimpleSequenceGroup
                total_tokens += item.get_total_tokens()
        
        return total_tokens
    
    def _calculate_budget_utilization(self, budget: SimpleSchedulingBudget) -> float:
        """计算预算利用率"""
        batch_utilization = budget.current_batch_size / budget.max_batch_size
        token_utilization = budget.current_tokens / budget.max_tokens
        
        return (batch_utilization + token_utilization) / 2
    
    def _update_stats(self, scheduled_groups: List[SimpleSequenceGroup], budget: SimpleSchedulingBudget):
        """更新统计信息"""
        self.stats["total_scheduled"] += len(scheduled_groups)
        
        # 计算平均批处理大小
        if self.stats["total_scheduled"] > 0:
            self.stats["avg_batch_size"] = (
                self.stats["avg_batch_size"] * (self.stats["total_scheduled"] - len(scheduled_groups)) +
                len(scheduled_groups)
            ) / self.stats["total_scheduled"]
        
        # 计算平均token数
        total_tokens = sum(group.get_total_tokens() for group in scheduled_groups)
        if len(scheduled_groups) > 0:
            self.stats["avg_tokens_per_batch"] = (
                self.stats["avg_tokens_per_batch"] * (self.stats["total_scheduled"] - len(scheduled_groups)) +
                total_tokens / len(scheduled_groups)
            ) / self.stats["total_scheduled"]
        
        # 计算调度效率
        self.stats["scheduling_efficiency"] = self._calculate_scheduling_efficiency()
    
    def _calculate_scheduling_efficiency(self) -> float:
        """计算调度效率"""
        if not self.waiting_queue and not self.running_queue:
            return 1.0
        
        # 简化的效率计算
        running_utilization = len(self.running_queue) / self.max_batch_size
        waiting_ratio = len(self.waiting_queue) / (len(self.waiting_queue) + len(self.running_queue))
        
        return running_utilization * (1 - waiting_ratio * 0.5)
    
    def _record_scheduling_history(self, output: SimpleSchedulerOutput):
        """记录调度历史"""
        history_entry = {
            "timestamp": time.time(),
            "scheduled_groups": len(output.scheduled_groups),
            "ignored_groups": len(output.ignored_groups),
            "budget_utilization": output.budget_utilization,
            "waiting_queue_size": len(self.waiting_queue),
            "running_queue_size": len(self.running_queue),
            "memory_utilization": self.memory_manager.get_memory_usage()["utilization"]
        }
        
        self.scheduling_history.append(history_entry)
        
        # 限制历史记录长度
        if len(self.scheduling_history) > 100:
            self.scheduling_history = self.scheduling_history[-100:]
    
    def get_stats(self) -> Dict[str, any]:
        """获取统计信息"""
        return {
            **self.stats,
            "waiting_queue_size": len(self.waiting_queue),
            "running_queue_size": len(self.running_queue),
            "memory_stats": self.memory_manager.get_stats()
        }
    
    def visualize_scheduling_performance(self):
        """可视化调度性能（教育用途）"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.scheduling_history:
                print("No scheduling history available for visualization.")
                return
            
            # 准备数据
            timestamps = [h["timestamp"] for h in self.scheduling_history]
            scheduled_groups = [h["scheduled_groups"] for h in self.scheduling_history]
            budget_utilization = [h["budget_utilization"] for h in self.scheduling_history]
            memory_utilization = [h["memory_utilization"] for h in self.scheduling_history]
            
            # 创建时间轴
            start_time = timestamps[0]
            relative_times = [(t - start_time) for t in timestamps]
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 调度组数
            ax1.plot(relative_times, scheduled_groups, 'b-', linewidth=2)
            ax1.set_title('Scheduled Groups Over Time')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Number of Groups')
            ax1.grid(True, alpha=0.3)
            
            # 2. 预算利用率
            ax2.plot(relative_times, budget_utilization, 'g-', linewidth=2)
            ax2.set_title('Budget Utilization Over Time')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Utilization Ratio')
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3)
            
            # 3. 内存利用率
            ax3.plot(relative_times, memory_utilization, 'r-', linewidth=2)
            ax3.set_title('Memory Utilization Over Time')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Utilization Ratio')
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
            
            # 4. 队列大小
            waiting_sizes = [h["waiting_queue_size"] for h in self.scheduling_history]
            running_sizes = [h["running_queue_size"] for h in self.scheduling_history]
            
            ax4.plot(relative_times, waiting_sizes, 'orange', label='Waiting Queue', linewidth=2)
            ax4.plot(relative_times, running_sizes, 'purple', label='Running Queue', linewidth=2)
            ax4.set_title('Queue Sizes Over Time')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Queue Size')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
    
    def print_debug_info(self):
        """打印调试信息"""
        print("=" * 60)
        print("SimpleScheduler Debug Info")
        print("=" * 60)
        
        stats = self.get_stats()
        print(f"Max Batch Size: {self.max_batch_size}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Enable Chunking: {self.enable_chunking}")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Waiting Queue: {stats['waiting_queue_size']} items")
        print(f"Running Queue: {stats['running_queue_size']} items")
        print(f"Total Scheduled: {stats['total_scheduled']}")
        print(f"Total Completed: {stats['total_completed']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Avg Batch Size: {stats['avg_batch_size']:.2f}")
        print(f"Avg Tokens per Batch: {stats['avg_tokens_per_batch']:.2f}")
        print(f"Scheduling Efficiency: {stats['scheduling_efficiency']:.2%}")
        
        print("\nMemory Stats:")
        memory_stats = stats['memory_stats']
        print(f"  Total Blocks: {memory_stats['total_blocks']}")
        print(f"  Used Blocks: {memory_stats['used_blocks']}")
        print(f"  Free Blocks: {memory_stats['free_blocks']}")
        print(f"  Utilization: {memory_stats['utilization']:.2%}")
        print(f"  Fragmentation: {memory_stats['fragmentation_ratio']:.2%}")
        
        print("\nQueue Contents:")
        print("Waiting Queue:")
        for i, item in enumerate(self.waiting_queue[:5]):  # 只显示前5个
            if isinstance(item, SimpleSequence):
                print(f"  {i}: {item.seq_id} ({len(item.token_ids)} tokens)")
            else:
                print(f"  {i}: {item.group_id} ({len(item)} sequences)")
        
        if len(self.waiting_queue) > 5:
            print(f"  ... and {len(self.waiting_queue) - 5} more items")
        
        print("Running Queue:")
        for i, item in enumerate(self.running_queue[:5]):  # 只显示前5个
            if isinstance(item, SimpleSequence):
                print(f"  {i}: {item.seq_id} ({item.status.value})")
            else:
                print(f"  {i}: {item.group_id} ({len(item.get_running_sequences())} running)")
        
        if len(self.running_queue) > 5:
            print(f"  ... and {len(self.running_queue) - 5} more items")
        
        print("=" * 60)
```

## 4. 简化的引擎实现

### 4.1 简化的引擎核心

```python
# simple_engine.py
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from simple_config import SimpleVllmConfig
from simple_scheduler import SimpleScheduler, SimpleSchedulerOutput
from simple_sequence import SimpleSequence, SimpleSequenceGroup, SequenceStatus
from simple_paged_attention import SimplePagedAttention

@dataclass
class SimpleEngineOutput:
    """简化的引擎输出"""
    sequences: List[SimpleSequence]
    total_time: float
    tokens_per_second: float
    memory_usage: Dict[str, Any]

class SimpleEngine:
    """简化的LLM引擎"""
    
    def __init__(self, config: SimpleVllmConfig):
        self.config = config
        
        # 初始化组件
        self.scheduler = SimpleScheduler(
            max_batch_size=config.scheduler_config.max_batch_size,
            max_tokens=config.scheduler_config.max_tokens,
            enable_chunking=config.scheduler_config.enable_chunking,
            chunk_size=config.scheduler_config.chunk_size
        )
        
        # 内存管理
        self.memory_manager = self.scheduler.memory_manager
        
        # 模型（简化版）
        self.model = self._create_dummy_model()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "avg_tokens_per_second": 0.0,
            "engine_efficiency": 0.0
        }
        
        # 性能监控
        self.performance_history = []
    
    def _create_dummy_model(self):
        """创建虚拟模型（用于演示）"""
        return DummyModel(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            device=self.config.model_config.device
        )
    
    def add_request(self, prompt: str, **kwargs) -> str:
        """添加推理请求"""
        
        # 创建序列
        seq_id = f"req_{int(time.time() * 1000)}_{self.stats['total_requests']}"
        
        # 虚拟tokenization（简化版）
        token_ids = self._dummy_tokenize(prompt)
        
        # 创建序列
        sequence = SimpleSequence(
            seq_id=seq_id,
            prompt=prompt,
            token_ids=token_ids,
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_tokens', 100)
        )
        
        # 添加到调度器
        self.scheduler.add_sequence(sequence)
        
        # 更新统计
        self.stats["total_requests"] += 1
        
        return seq_id
    
    def step(self) -> SimpleEngineOutput:
        """执行一个推理步骤"""
        
        start_time = time.time()
        
        # 1. 调度
        scheduler_output = self.scheduler.schedule_step()
        
        # 2. 执行推理
        if scheduler_output.scheduled_groups:
            self._execute_inference(scheduler_output.scheduled_groups)
        
        # 3. 收集结果
        output_sequences = self._collect_output_sequences()
        
        # 4. 计算性能指标
        step_time = time.time() - start_time
        total_tokens = sum(seq.get_num_generated_tokens() for seq in output_sequences)
        tokens_per_second = total_tokens / step_time if step_time > 0 else 0
        
        # 5. 获取内存使用情况
        memory_usage = self.memory_manager.get_memory_usage()
        
        # 6. 更新统计
        self._update_engine_stats(output_sequences, step_time, tokens_per_second)
        
        # 7. 记录性能历史
        self._record_performance_history(step_time, tokens_per_second, memory_usage)
        
        return SimpleEngineOutput(
            sequences=output_sequences,
            total_time=step_time,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_usage
        )
    
    def _execute_inference(self, sequence_groups: List[SimpleSequenceGroup]):
        """执行推理（简化版）"""
        
        for group in sequence_groups:
            for sequence in group.get_running_sequences():
                if not sequence.is_finished():
                    # 模拟推理过程
                    self._simulate_sequence_inference(sequence)
    
    def _simulate_sequence_inference(self, sequence: SimpleSequence):
        """模拟序列推理过程"""
        
        # 获取KV缓存
        keys, values = self.memory_manager.retrieve_kv_cache(
            sequence.seq_id, 
            0, 
            len(sequence.token_ids)
        )
        
        # 模拟逐步生成
        for step in range(min(sequence.max_tokens, 10)):  # 限制生成步数用于演示
            # 创建虚拟query
            query = torch.randn(1, 64)  # 简化的query
            
            # 计算注意力
            if keys.numel() > 0:
                attention_output = self.memory_manager.compute_attention(
                    query, keys, values, scale=1.0
                )
            else:
                attention_output = query
            
            # 模拟采样
            token_id = self._dummy_sample(attention_output)
            token_text = f"[token_{token_id}]"
            
            # 添加到序列
            sequence.add_generated_token(token_id, token_text)
            
            # 更新KV缓存
            new_key = torch.randn(64)
            new_value = torch.randn(64)
            self.memory_manager.store_kv_cache(
                sequence.seq_id,
                [token_id],
                new_key.unsqueeze(0),
                new_value.unsqueeze(0)
            )
            
            # 简化的停止条件
            if step >= 5 or token_id == 2:  # 假设token 2是EOS
                sequence.finish_generation()
                break
    
    def _collect_output_sequences(self) -> List[SimpleSequence]:
        """收集输出序列"""
        output_sequences = []
        
        # 从运行队列中收集已完成的序列
        for item in list(self.scheduler.running_queue):
            if isinstance(item, SimpleSequence) and item.is_finished():
                output_sequences.append(item)
                self.scheduler.running_queue.remove(item)
            elif isinstance(item, SimpleSequenceGroup):
                finished_seqs = item.get_finished_sequences()
                if finished_seqs:
                    output_sequences.extend(finished_seqs)
                    if len(finished_seqs) == len(item):
                        self.scheduler.running_queue.remove(item)
        
        return output_sequences
    
    def _dummy_tokenize(self, text: str) -> List[int]:
        """虚拟tokenization"""
        # 简化的tokenization，实际应用中应该使用真正的tokenizer
        return [ord(c) % 1000 for c in text[:100]]  # 限制长度并简化
    
    def _dummy_sample(self, logits: torch.Tensor) -> int:
        """虚拟采样"""
        # 简化的采样逻辑
        if logits.numel() == 0:
            return 1
        
        # 简单的argmax采样
        return int(torch.argmax(logits).item())
    
    def _update_engine_stats(self, sequences: List[SimpleSequence], 
                           step_time: float, tokens_per_second: float):
        """更新引擎统计"""
        total_tokens = sum(seq.get_num_generated_tokens() for seq in sequences)
        
        self.stats["total_tokens"] += total_tokens
        self.stats["total_time"] += step_time
        
        if self.stats["total_time"] > 0:
            self.stats["avg_tokens_per_second"] = self.stats["total_tokens"] / self.stats["total_time"]
        
        # 计算引擎效率
        self.stats["engine_efficiency"] = self._calculate_engine_efficiency()
    
    def _calculate_engine_efficiency(self) -> float:
        """计算引擎效率"""
        if self.stats["total_requests"] == 0:
            return 0.0
        
        # 简化的效率计算
        scheduling_efficiency = self.scheduler.stats["scheduling_efficiency"]
        memory_efficiency = self.memory_manager.get_stats()["efficiency"]
        
        return (scheduling_efficiency + memory_efficiency) / 2
    
    def _record_performance_history(self, step_time: float, 
                                    tokens_per_second: float, 
                                    memory_usage: Dict[str, Any]):
        """记录性能历史"""
        history_entry = {
            "timestamp": time.time(),
            "step_time": step_time,
            "tokens_per_second": tokens_per_second,
            "memory_utilization": memory_usage["utilization"],
            "waiting_queue_size": len(self.scheduler.waiting_queue),
            "running_queue_size": len(self.scheduler.running_queue),
            "scheduled_groups": len(self.scheduler.scheduling_history[-1]["scheduled_groups"]) if self.scheduler.scheduling_history else 0
        }
        
        self.performance_history.append(history_entry)
        
        # 限制历史记录长度
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def run_until_completion(self, max_steps: int = 100) -> List[SimpleSequence]:
        """运行直到所有请求完成"""
        
        all_outputs = []
        
        for step in range(max_steps):
            if not self.scheduler.has_unfinished_requests():
                break
            
            output = self.step()
            all_outputs.extend(output.sequences)
            
            # 打印进度
            if step % 10 == 0:
                print(f"Step {step}: {len(output.sequences)} sequences completed, "
                      f"{output.tokens_per_second:.1f} tokens/sec")
        
        return all_outputs
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            **self.stats,
            "scheduler_stats": self.scheduler.get_stats(),
            "memory_stats": self.memory_manager.get_stats()
        }
    
    def visualize_performance(self):
        """可视化性能（教育用途）"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.performance_history:
                print("No performance history available for visualization.")
                return
            
            # 准备数据
            timestamps = [h["timestamp"] for h in self.performance_history]
            step_times = [h["step_time"] for h in self.performance_history]
            tokens_per_second = [h["tokens_per_second"] for h in self.performance_history]
            memory_utilization = [h["memory_utilization"] for h in self.performance_history]
            
            # 创建时间轴
            start_time = timestamps[0]
            relative_times = [(t - start_time) for t in timestamps]
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 步骤时间
            ax1.plot(relative_times, step_times, 'b-', linewidth=2)
            ax1.set_title('Step Time Over Time')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Step Time (seconds)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Tokens per second
            ax2.plot(relative_times, tokens_per_second, 'g-', linewidth=2)
            ax2.set_title('Tokens Per Second Over Time')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Tokens/Second')
            ax2.grid(True, alpha=0.3)
            
            # 3. 内存利用率
            ax3.plot(relative_times, memory_utilization, 'r-', linewidth=2)
            ax3.set_title('Memory Utilization Over Time')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Utilization Ratio')
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
            
            # 4. 队列大小
            waiting_sizes = [h["waiting_queue_size"] for h in self.performance_history]
            running_sizes = [h["running_queue_size"] for h in self.performance_history]
            
            ax4.plot(relative_times, waiting_sizes, 'orange', label='Waiting', linewidth=2)
            ax4.plot(relative_times, running_sizes, 'purple', label='Running', linewidth=2)
            ax4.set_title('Queue Sizes Over Time')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Queue Size')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
    
    def print_debug_info(self):
        """打印调试信息"""
        print("=" * 70)
        print("SimpleEngine Debug Info")
        print("=" * 70)
        
        stats = self.get_stats()
        print(f"Config:")
        print(f"  Model: {self.config.model_config.model_name}")
        print(f"  Device: {self.config.model_config.device}")
        print(f"  Max Batch Size: {self.config.scheduler_config.max_batch_size}")
        print(f"  Max Tokens: {self.config.scheduler_config.max_tokens}")
        print(f"  Enable Chunking: {self.config.scheduler_config.enable_chunking}")
        
        print(f"\nEngine Stats:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Total Tokens: {stats['total_tokens']}")
        print(f"  Total Time: {stats['total_time']:.2f}s")
        print(f"  Avg Tokens/Second: {stats['avg_tokens_per_second']:.2f}")
        print(f"  Engine Efficiency: {stats['engine_efficiency']:.2%}")
        
        print(f"\nScheduler Stats:")
        scheduler_stats = stats['scheduler_stats']
        print(f"  Total Scheduled: {scheduler_stats['total_scheduled']}")
        print(f"  Total Completed: {scheduler_stats['total_completed']}")
        print(f"  Total Chunks: {scheduler_stats['total_chunks']}")
        print(f"  Avg Batch Size: {scheduler_stats['avg_batch_size']:.2f}")
        print(f"  Scheduling Efficiency: {scheduler_stats['scheduling_efficiency']:.2%}")
        print(f"  Waiting Queue: {scheduler_stats['waiting_queue_size']}")
        print(f"  Running Queue: {scheduler_stats['running_queue_size']}")
        
        print(f"\nMemory Stats:")
        memory_stats = stats['memory_stats']
        print(f"  Total Blocks: {memory_stats['total_blocks']}")
        print(f"  Used Blocks: {memory_stats['used_blocks']}")
        print(f"  Free Blocks: {memory_stats['free_blocks']}")
        print(f"  Utilization: {memory_stats['utilization']:.2%}")
        print(f"  Fragmentation: {memory_stats['fragmentation_ratio']:.2%}")
        print(f"  Efficiency: {memory_stats['efficiency']:.2%}")
        
        print("=" * 70)

class DummyModel:
    """虚拟模型（用于演示）"""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, 
                 num_heads: int, device: str):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device
        
        # 创建虚拟参数
        self.embed_tokens = torch.randn(vocab_size, hidden_size, device=device)
        self.layers = [torch.nn.Linear(hidden_size, hidden_size, device=device) 
                       for _ in range(num_layers)]
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, device=device)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """简化的前向传播"""
        # 词嵌入
        hidden = self.embed_tokens[input_ids]
        
        # Transformer层
        for layer in self.layers:
            hidden = layer(hidden)
            hidden = torch.relu(hidden)
        
        # 语言模型头
        logits = self.lm_head(hidden)
        
        return logits
```

### 4.2 简化的用户接口

```python
# simple_llm.py
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from simple_engine import SimpleEngine, SimpleEngineOutput
from simple_config import SimpleVllmConfig
from simple_sequence import SimpleSequence

@dataclass
class SimpleLLMOutput:
    """简化的LLM输出"""
    request_id: str
    prompt: str
    generated_text: str
    generated_tokens: List[int]
    generation_time: float
    tokens_per_second: float

class SimpleLLM:
    """简化的LLM接口"""
    
    def __init__(self, config: Optional[SimpleVllmConfig] = None):
        if config is None:
            config = SimpleVllmConfig()
        
        self.config = config
        self.engine = SimpleEngine(config)
        
        # 请求管理
        self.request_map: Dict[str, SimpleSequence] = {}
        self.completed_requests: Dict[str, SimpleLLMOutput] = {}
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_tokens_per_second": 0.0
        }
    
    def generate(self, prompts: List[str], **kwargs) -> List[SimpleLLMOutput]:
        """生成文本"""
        
        # 添加所有请求
        request_ids = []
        for prompt in prompts:
            request_id = self.engine.add_request(prompt, **kwargs)
            request_ids.append(request_id)
        
        # 运行直到完成
        start_time = time.time()
        all_sequences = self.engine.run_until_completion()
        total_time = time.time() - start_time
        
        # 收集结果
        outputs = []
        for request_id in request_ids:
            if request_id in self.completed_requests:
                outputs.append(self.completed_requests[request_id])
            else:
                # 从引擎输出中查找
                for seq in all_sequences:
                    if seq.seq_id == request_id:
                        output = self._create_llm_output(seq)
                        outputs.append(output)
                        self.completed_requests[request_id] = output
                        break
        
        # 更新统计
        self._update_llm_stats(outputs, total_time)
        
        return outputs
    
    def generate_async(self, prompt: str, **kwargs) -> str:
        """异步生成（简化版）"""
        request_id = self.engine.add_request(prompt, **kwargs)
        return request_id
    
    def get_output(self, request_id: str) -> Optional[SimpleLLMOutput]:
        """获取请求输出"""
        if request_id in self.completed_requests:
            return self.completed_requests[request_id]
        
        # 检查是否在引擎中完成
        for seq in self.engine.scheduler.running_queue:
            if isinstance(seq, SimpleSequence) and seq.seq_id == request_id:
                if seq.is_finished():
                    output = self._create_llm_output(seq)
                    self.completed_requests[request_id] = output
                    return output
        
        return None
    
    def step(self) -> List[SimpleLLMOutput]:
        """执行一个步骤"""
        engine_output = self.engine.step()
        
        # 转换为LLM输出
        outputs = []
        for seq in engine_output.sequences:
            if seq.seq_id not in self.completed_requests:
                output = self._create_llm_output(seq)
                self.completed_requests[seq.seq_id] = output
                outputs.append(output)
        
        return outputs
    
    def has_unfinished_requests(self) -> bool:
        """检查是否有未完成的请求"""
        return self.engine.scheduler.has_unfinished_requests()
    
    def get_num_unfinished_requests(self) -> int:
        """获取未完成的请求数量"""
        return len(self.engine.scheduler.waiting_queue) + len(self.engine.scheduler.running_queue)
    
    def _create_llm_output(self, sequence: SimpleSequence) -> SimpleLLMOutput:
        """创建LLM输出"""
        return SimpleLLMOutput(
            request_id=sequence.seq_id,
            prompt=sequence.prompt,
            generated_text=sequence.generated_text,
            generated_tokens=sequence.generated_token_ids,
            generation_time=sequence.get_generation_time(),
            tokens_per_second=sequence.get_tokens_per_second()
        )
    
    def _update_llm_stats(self, outputs: List[SimpleLLMOutput], total_time: float):
        """更新LLM统计"""
        self.stats["total_requests"] += len(outputs)
        self.stats["total_tokens"] += sum(len(output.generated_tokens) for output in outputs)
        self.stats["total_time"] += total_time
        
        if len(outputs) > 0:
            self.stats["avg_generation_time"] = (
                self.stats["avg_generation_time"] * (self.stats["total_requests"] - len(outputs)) +
                sum(output.generation_time for output in outputs)
            ) / self.stats["total_requests"]
            
            self.stats["avg_tokens_per_second"] = (
                self.stats["avg_tokens_per_second"] * (self.stats["total_requests"] - len(outputs)) +
                sum(output.tokens_per_second for output in outputs)
            ) / self.stats["total_requests"]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "engine_stats": self.engine.get_stats()
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        print("=" * 60)
        print("SimpleLLM Statistics")
        print("=" * 60)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Avg Generation Time: {stats['avg_generation_time']:.3f}s")
        print(f"Avg Tokens/Second: {stats['avg_tokens_per_second']:.2f}")
        
        if stats['total_requests'] > 0:
            print(f"Unfinished Requests: {self.get_num_unfinished_requests()}")
        
        print("=" * 60)
    
    def visualize_performance(self):
        """可视化性能"""
        self.engine.visualize_performance()
        self.engine.scheduler.visualize_scheduling_performance()
        self.engine.memory_manager.visualize_memory_layout()
    
    def print_debug_info(self):
        """打印调试信息"""
        print("=" * 80)
        print("SimpleLLM Debug Info")
        print("=" * 80)
        
        self.engine.print_debug_info()
        
        print(f"\nCompleted Requests: {len(self.completed_requests)}")
        print(f"Request Map: {list(self.request_map.keys())[:10]}")  # 显示前10个
        
        if len(self.completed_requests) > 0:
            print(f"\nRecent Completed Requests:")
            for i, (req_id, output) in enumerate(list(self.completed_requests.items())[-5:]):
                print(f"  {i+1}. {req_id}: {len(output.generated_tokens)} tokens, "
                      f"{output.tokens_per_second:.1f} tokens/sec")
        
        print("=" * 80)
```

## 5. 使用示例和教程

### 5.1 基本使用示例

```python
# examples/basic_usage.py
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_llm import SimpleLLM
from simple_config import SimpleVllmConfig

def main():
    print("=== vLLM教育版演示 ===")
    
    # 1. 创建配置
    config = SimpleVllmConfig()
    print(f"配置: {config}")
    
    # 2. 初始化LLM
    print("\n1. 初始化SimpleLLM...")
    llm = SimpleLLM(config)
    print("✓ SimpleLLM初始化完成")
    
    # 3. 准备提示词
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In the future, AI will",
        "Machine learning is",
        "The weather today is"
    ]
    
    print(f"\n2. 准备生成 {len(prompts)} 个提示词...")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    
    # 4. 生成文本
    print("\n3. 开始生成文本...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, max_tokens=50)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 5. 显示结果
    print(f"\n4. 生成结果 (总时间: {total_time:.2f}s):")
    print("-" * 60)
    
    for i, output in enumerate(outputs):
        print(f"请求 {i+1}:")
        print(f"  提示词: {output.prompt}")
        print(f"  生成文本: {output.generated_text}")
        print(f"  生成token数: {len(output.generated_tokens)}")
        print(f"  生成时间: {output.generation_time:.3f}s")
        print(f"  速度: {output.tokens_per_second:.1f} tokens/sec")
        print()
    
    # 6. 显示统计信息
    print("5. 统计信息:")
    llm.print_stats()
    
    # 7. 可视化性能
    print("\n6. 性能可视化...")
    llm.visualize_performance()
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    import time
    main()
```

### 5.2 分步教程

```python
# tutorials/tutorial_step_by_step.py
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_config import SimpleVllmConfig
from simple_paged_attention import SimplePagedAttention
from simple_scheduler import SimpleScheduler
from simple_engine import SimpleEngine
from simple_llm import SimpleLLM

def tutorial_paged_attention():
    """教程1: PagedAttention原理"""
    print("=" * 60)
    print("教程1: PagedAttention原理")
    print("=" * 60)
    
    # 1. 创建PagedAttention实例
    paged_attn = SimplePagedAttention(block_size=4, max_blocks=10)
    print(f"✓ 创建PagedAttention: block_size=4, max_blocks=10")
    
    # 2. 模拟序列
    seq1_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10个token，需要3个块
    seq2_tokens = [11, 12, 13, 14, 15]              # 5个token，需要2个块
    
    print(f"\n序列1: {seq1_tokens} (需要3个块)")
    print(f"序列2: {seq2_tokens} (需要2个块)")
    
    # 3. 分配块
    print("\n分配内存块...")
    seq1_blocks = paged_attn.allocate_blocks("seq1", len(seq1_tokens))
    seq2_blocks = paged_attn.allocate_blocks("seq2", len(seq2_tokens))
    
    print(f"✓ 序列1分配块: {seq1_blocks}")
    print(f"✓ 序列2分配块: {seq2_blocks}")
    
    # 4. 存储KV缓存
    print("\n存储KV缓存...")
    import torch
    
    seq1_keys = torch.randn(len(seq1_tokens), 64)
    seq1_values = torch.randn(len(seq1_tokens), 64)
    seq2_keys = torch.randn(len(seq2_tokens), 64)
    seq2_values = torch.randn(len(seq2_tokens), 64)
    
    paged_attn.store_kv_cache("seq1", seq1_tokens, seq1_keys, seq1_values)
    paged_attn.store_kv_cache("seq2", seq2_tokens, seq2_keys, seq2_values)
    
    print("✓ KV缓存存储完成")
    
    # 5. 检索KV缓存
    print("\n检索KV缓存...")
    retrieved_keys, retrieved_values = paged_attn.retrieve_kv_cache("seq1", 2, 8)
    print(f"✓ 检索到keys: {retrieved_keys.shape}")
    print(f"✓ 检索到values: {retrieved_values.shape}")
    
    # 6. 计算注意力
    print("\n计算注意力...")
    query = torch.randn(1, 64)
    attention_output = paged_attn.compute_attention(query, retrieved_keys, retrieved_values)
    print(f"✓ 注意力输出: {attention_output.shape}")
    
    # 7. 显示内存使用情况
    print("\n内存使用情况:")
    memory_stats = paged_attn.get_memory_usage()
    for key, value in memory_stats.items():
        print(f"  {key}: {value}")
    
    # 8. 可视化内存布局
    print("\n可视化内存布局...")
    paged_attn.visualize_memory_layout()
    
    # 9. 释放内存
    print("\n释放内存...")
    paged_attn.free_blocks("seq1")
    paged_attn.free_blocks("seq2")
    
    print("✓ 内存释放完成")
    paged_attn.print_debug_info()

def tutorial_scheduler():
    """教程2: 调度器原理"""
    print("=" * 60)
    print("教程2: 调度器原理")
    print("=" * 60)
    
    # 1. 创建调度器
    scheduler = SimpleScheduler(max_batch_size=3, max_tokens=100)
    print(f"✓ 创建调度器: max_batch_size=3, max_tokens=100")
    
    # 2. 创建序列
    from simple_sequence import SimpleSequence
    
    sequences = []
    prompts = [
        "Short prompt",
        "This is a medium length prompt that requires more processing",
        "This is a very long prompt that definitely needs chunking to process efficiently within the memory constraints",
        "Another short prompt",
        "Medium length prompt for testing"
    ]
    
    print(f"\n创建 {len(prompts)} 个序列:")
    for i, prompt in enumerate(prompts):
        seq = SimpleSequence(
            seq_id=f"seq_{i}",
            prompt=prompt,
            token_ids=list(range(len(prompt.split()))),  # 简化的tokenization
            max_tokens=20
        )
        sequences.append(seq)
        print(f"  {i+1}. {prompt} ({len(seq.token_ids)} tokens)")
    
    # 3. 添加序列到调度器
    print("\n添加序列到调度器...")
    for seq in sequences:
        scheduler.add_sequence(seq)
    print(f"✓ 已添加 {len(sequences)} 个序列到等待队列")
    
    # 4. 执行调度步骤
    print("\n执行调度步骤...")
    
    for step in range(5):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 调度
        output = scheduler.schedule_step()
        
        print(f"调度组数: {len(output.scheduled_groups)}")
        print(f"预算利用率: {output.budget_utilization:.2%}")
        print(f"等待队列: {len(output.ignored_groups)}")
        
        # 模拟推理完成
        for group in output.scheduled_groups:
            for seq in group.sequences:
                if seq.is_running():
                    # 模拟生成一些token
                    seq.add_generated_token(100 + step, f"[token_{100 + step}]")
                    if step >= 2:  # 模拟在第3步完成
                        seq.finish_generation()
        
        # 显示当前状态
        waiting_count = len(scheduler.waiting_queue)
        running_count = len(scheduler.running_queue)
        print(f"等待队列: {waiting_count}, 运行队列: {running_count}")
        
        # 如果没有等待的序列，停止调度
        if waiting_count == 0:
            print("所有序列已完成，停止调度")
            break
    
    # 5. 显示统计信息
    print("\n调度统计信息:")
    scheduler.print_debug_info()
    
    # 6. 可视化调度性能
    print("\n可视化调度性能...")
    scheduler.visualize_scheduling_performance()

def tutorial_engine():
    """教程3: 引擎原理"""
    print("=" * 60)
    print("教程3: 引擎原理")
    print("=" * 60)
    
    # 1. 创建引擎
    config = SimpleVllmConfig()
    engine = SimpleEngine(config)
    print(f"✓ 创建SimpleEngine")
    
    # 2. 添加请求
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "How do neural networks work?"
    ]
    
    print(f"\n添加 {len(prompts)} 个请求...")
    request_ids = []
    for prompt in prompts:
        request_id = engine.add_request(prompt, max_tokens=30)
        request_ids.append(request_id)
        print(f"✓ 添加请求: {request_id} - {prompt}")
    
    # 3. 执行推理步骤
    print(f"\n执行推理步骤...")
    
    for step in range(10):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 执行步骤
        start_time = time.time()
        output = engine.step()
        step_time = time.time() - start_time
        
        print(f"步骤时间: {step_time:.3f}s")
        print(f"输出序列数: {len(output.sequences)}")
        print(f"Tokens/Second: {output.tokens_per_second:.1f}")
        print(f"内存利用率: {output.memory_usage['utilization']:.2%}")
        
        # 显示新生成的序列
        if output.sequences:
            print("新生成的序列:")
            for seq in output.sequences:
                print(f"  {seq.seq_id}: {seq.generated_text}")
        
        # 检查是否完成
        if not engine.has_unfinished_requests():
            print("所有请求已完成！")
            break
    
    # 4. 显示最终统计
    print("\n最终统计信息:")
    engine.print_debug_info()
    
    # 5. 可视化性能
    print("\n可视化引擎性能...")
    engine.visualize_performance()

def tutorial_complete_workflow():
    """教程4: 完整工作流程"""
    print("=" * 60)
    print("教程4: 完整工作流程")
    print("=" * 60)
    
    # 1. 创建LLM
    print("1. 创建SimpleLLM...")
    llm = SimpleLLM()
    print("✓ SimpleLLM创建完成")
    
    # 2. 批量生成
    print("\n2. 批量生成文本...")
    
    prompts = [
        "The future of technology",
        "Climate change solutions",
        "Space exploration challenges",
        "Artificial intelligence ethics",
        "Renewable energy innovation"
    ]
    
    print(f"生成 {len(prompts)} 个提示词的文本...")
    
    start_time = time.time()
    outputs = llm.generate(prompts, max_tokens=40, temperature=0.8)
    end_time = time.time()
    
    # 3. 显示结果
    print(f"\n生成完成 (总时间: {end_time - start_time:.2f}s):")
    print("-" * 60)
    
    for i, output in enumerate(outputs):
        print(f"结果 {i+1}:")
        print(f"  提示词: {output.prompt}")
        print(f"  生成文本: {output.generated_text}")
        print(f"  Token数: {len(output.generated_tokens)}")
        print(f"  生成时间: {output.generation_time:.3f}s")
        print(f"  速度: {output.tokens_per_second:.1f} tokens/sec")
        print()
    
    # 4. 显示统计
    print("统计信息:")
    llm.print_stats()
    
    # 5. 可视化
    print("\n可视化性能...")
    llm.visualize_performance()
    
    print("\n=== 教程完成 ===")

def main():
    """主函数"""
    print("vLLM教育版 - 分步教程")
    print("=" * 80)
    
    tutorials = [
        ("PagedAttention原理", tutorial_paged_attention),
        ("调度器原理", tutorial_scheduler),
        ("引擎原理", tutorial_engine),
        ("完整工作流程", tutorial_complete_workflow)
    ]
    
    print("可用教程:")
    for i, (name, _) in enumerate(tutorials):
        print(f"{i+1}. {name}")
    
    try:
        choice = input("\n选择教程 (1-4): ").strip()
        tutorial_index = int(choice) - 1
        
        if 0 <= tutorial_index < len(tutorials):
            _, tutorial_func = tutorials[tutorial_index]
            tutorial_func()
        else:
            print("无效选择")
    
    except (ValueError, KeyboardInterrupt):
        print("\n程序退出")

if __name__ == "__main__":
    main()
```

### 5.3 性能对比教程

```python
# tutorials/performance_comparison.py
import sys
import os
import time
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_llm import SimpleLLM
from simple_config import SimpleVllmConfig

def compare_batching_strategies():
    """对比批处理策略"""
    print("=" * 60)
    print("批处理策略对比")
    print("=" * 60)
    
    # 测试不同的批处理大小
    batch_sizes = [1, 2, 4, 8, 16]
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n测试批处理大小: {batch_size}")
        
        # 创建配置
        config = SimpleVllmConfig()
        config.scheduler_config.max_batch_size = batch_size
        
        # 创建LLM
        llm = SimpleLLM(config)
        
        # 准备测试数据
        prompts = ["Test prompt " + str(i) for i in range(20)]
        
        # 测试
        start_time = time.time()
        outputs = llm.generate(prompts, max_tokens=20)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = sum(len(output.generated_tokens) for output in outputs)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        results[batch_size] = {
            "total_time": total_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "efficiency": tokens_per_second / batch_size
        }
        
        print(f"  总时间: {total_time:.2f}s")
        print(f"  总token数: {total_tokens}")
        print(f"  Tokens/Second: {tokens_per_second:.1f}")
        print(f"  效率: {tokens_per_second / batch_size:.1f}")
    
    # 可视化结果
    visualize_batching_comparison(results)
    
    return results

def compare_memory_strategies():
    """对比内存策略"""
    print("=" * 60)
    print("内存策略对比")
    print("=" * 60)
    
    # 测试不同的块大小
    block_sizes = [4, 8, 16, 32, 64]
    
    results = {}
    
    for block_size in block_sizes:
        print(f"\n测试块大小: {block_size}")
        
        # 创建配置
        config = SimpleVllmConfig()
        config.cache_config.block_size = block_size
        
        # 创建LLM
        llm = SimpleLLM(config)
        
        # 准备测试数据
        prompts = ["This is a test prompt for memory management " + str(i) for i in range(10)]
        
        # 测试
        start_time = time.time()
        outputs = llm.generate(prompts, max_tokens=30)
        end_time = time.time()
        
        total_time = end_time - start_time
        memory_stats = llm.engine.memory_manager.get_stats()
        
        results[block_size] = {
            "total_time": total_time,
            "memory_utilization": memory_stats["utilization"],
            "fragmentation": memory_stats["fragmentation_ratio"],
            "efficiency": memory_stats["efficiency"]
        }
        
        print(f"  总时间: {total_time:.2f}s")
        print(f"  内存利用率: {memory_stats['utilization']:.2%}")
        print(f"  碎片化: {memory_stats['fragmentation_ratio']:.2%}")
        print(f"  效率: {memory_stats['efficiency']:.2%}")
    
    # 可视化结果
    visualize_memory_comparison(results)
    
    return results

def compare_scheduling_strategies():
    """对比调度策略"""
    print("=" * 60)
    print("调度策略对比")
    print("=" * 60)
    
    # 测试不同的调度配置
    scheduling_configs = [
        {"name": "基础调度", "max_batch_size": 4, "enable_chunking": False},
        {"name": "连续批处理", "max_batch_size": 8, "enable_chunking": False},
        {"name": "分块调度", "max_batch_size": 8, "enable_chunking": True, "chunk_size": 32},
        {"name": "优化调度", "max_batch_size": 16, "enable_chunking": True, "chunk_size": 64}
    ]
    
    results = {}
    
    for config in scheduling_configs:
        print(f"\n测试调度策略: {config['name']}")
        
        # 创建配置
        vllm_config = SimpleVllmConfig()
        vllm_config.scheduler_config.max_batch_size = config["max_batch_size"]
        vllm_config.scheduler_config.enable_chunking = config["enable_chunking"]
        if "chunk_size" in config:
            vllm_config.scheduler_config.chunk_size = config["chunk_size"]
        
        # 创建LLM
        llm = SimpleLLM(vllm_config)
        
        # 准备测试数据
        prompts = [
            "Short prompt " + str(i) for i in range(5)
        ] + [
            "This is a medium length prompt for testing scheduling performance " + str(i) for i in range(5)
        ] + [
            "This is a very long prompt that requires chunking to process efficiently " + str(i) for i in range(5)
        ]
        
        # 测试
        start_time = time.time()
        outputs = llm.generate(prompts, max_tokens=25)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = sum(len(output.generated_tokens) for output in outputs)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        scheduler_stats = llm.engine.scheduler.get_stats()
        
        results[config["name"]] = {
            "total_time": total_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "scheduling_efficiency": scheduler_stats["scheduling_efficiency"],
            "total_chunks": scheduler_stats["total_chunks"],
            "avg_batch_size": scheduler_stats["avg_batch_size"]
        }
        
        print(f"  总时间: {total_time:.2f}s")
        print(f"  Tokens/Second: {tokens_per_second:.1f}")
        print(f"  调度效率: {scheduler_stats['scheduling_efficiency']:.2%}")
        print(f"  总块数: {scheduler_stats['total_chunks']}")
        print(f"  平均批处理大小: {scheduler_stats['avg_batch_size']:.2f}")
    
    # 可视化结果
    visualize_scheduling_comparison(results)
    
    return results

def visualize_batching_comparison(results):
    """可视化批处理对比结果"""
    
    batch_sizes = list(results.keys())
    tokens_per_second = [results[bs]["tokens_per_second"] for bs in batch_sizes]
    efficiency = [results[bs]["efficiency"] for bs in batch_sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Tokens per second
    ax1.bar(batch_sizes, tokens_per_second, color='skyblue')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Tokens per Second')
    ax1.set_title('Throughput vs Batch Size')
    ax1.grid(True, alpha=0.3)
    
    # Efficiency
    ax2.bar(batch_sizes, efficiency, color='lightgreen')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Efficiency (Tokens/Second/Batch)')
    ax2.set_title('Efficiency vs Batch Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_memory_comparison(results):
    """可视化内存对比结果"""
    
    block_sizes = list(results.keys())
    memory_utilization = [results[bs]["memory_utilization"] for bs in block_sizes]
    fragmentation = [results[bs]["fragmentation"] for bs in block_sizes]
    efficiency = [results[bs]["efficiency"] for bs in block_sizes]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Memory utilization
    ax1.bar(block_sizes, memory_utilization, color='lightcoral')
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Memory Utilization')
    ax1.set_title('Memory Utilization vs Block Size')
    ax1.grid(True, alpha=0.3)
    
    # Fragmentation
    ax2.bar(block_sizes, fragmentation, color='orange')
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Fragmentation Ratio')
    ax2.set_title('Fragmentation vs Block Size')
    ax2.grid(True, alpha=0.3)
    
    # Efficiency
    ax3.bar(block_sizes, efficiency, color='lightgreen')
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Efficiency')
    ax3.set_title('Efficiency vs Block Size')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_scheduling_comparison(results):
    """可视化调度对比结果"""
    
    strategies = list(results.keys())
    tokens_per_second = [results[s]["tokens_per_second"] for s in strategies]
    scheduling_efficiency = [results[s]["scheduling_efficiency"] for s in strategies]
    total_chunks = [results[s]["total_chunks"] for s in strategies]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Tokens per second
    ax1.bar(strategies, tokens_per_second, color='skyblue')
    ax1.set_ylabel('Tokens per Second')
    ax1.set_title('Throughput vs Strategy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Scheduling efficiency
    ax2.bar(strategies, scheduling_efficiency, color='lightgreen')
    ax2.set_ylabel('Scheduling Efficiency')
    ax2.set_title('Scheduling Efficiency vs Strategy')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Total chunks
    ax3.bar(strategies, total_chunks, color='orange')
    ax3.set_ylabel('Total Chunks')
    ax3.set_title('Chunking Overhead vs Strategy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 综合对比
    metrics = ['tokens_per_second', 'scheduling_efficiency', 'total_chunks']
    values = []
    for strategy in strategies:
        strategy_values = [
            results[strategy]["tokens_per_second"] / max(results[s]["tokens_per_second"] for s in strategies),
            results[strategy]["scheduling_efficiency"],
            1 - results[strategy]["total_chunks"] / max(results[s]["total_chunks"] for s in strategies)
        ]
        values.append(strategy_values)
    
    im = ax4.imshow(values, cmap='RdYlGn', aspect='auto')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels(strategies, rotation=45)
    ax4.set_yticks(range(len(metrics)))
    ax4.set_yticklabels(['Throughput', 'Efficiency', 'Low Overhead'])
    ax4.set_title('Overall Performance Heatmap')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("vLLM教育版 - 性能对比教程")
    print("=" * 80)
    
    comparisons = [
        ("批处理策略对比", compare_batching_strategies),
        ("内存策略对比", compare_memory_strategies),
        ("调度策略对比", compare_scheduling_strategies)
    ]
    
    print("可用对比:")
    for i, (name, _) in enumerate(comparisons):
        print(f"{i+1}. {name}")
    
    try:
        choice = input("\n选择对比 (1-3): ").strip()
        comparison_index = int(choice) - 1
        
        if 0 <= comparison_index < len(comparisons):
            _, comparison_func = comparisons[comparison_index]
            comparison_func()
        else:
            print("无效选择")
    
    except (ValueError, KeyboardInterrupt):
        print("\n程序退出")

if __name__ == "__main__":
    main()
```

## 6. 总结和展望

### 6.1 教育版特色

1. **简化的实现**:
   - 移除了复杂的CUDA编程和分布式功能
   - 使用Python原生实现，便于理解
   - 保留了核心概念和思想

2. **丰富的可视化**:
   - 内存布局可视化
   - 调度性能可视化
   - 性能对比图表
   - 实时监控面板

3. **完整的教学材料**:
   - 分步教程
   - 概念解释
   - 代码注释
   - 练习题

4. **可扩展的架构**:
   - 模块化设计
   - 清晰的接口
   - 易于修改和实验

### 6.2 学习路径建议

**初学者**:
1. 从基本使用示例开始
2. 理解PagedAttention的概念
3. 学习调度器的原理
4. 完成分步教程

**进阶学习者**:
1. 深入研究内存管理
2. 分析性能优化策略
3. 进行性能对比实验
4. 尝试改进实现

**研究者**:
1. 对比教育版与完整版的差异
2. 分析优化策略的效果
3. 实现新的调度算法
4. 扩展功能到分布式环境

### 6.3 实践建议

1. **动手实践**:
   - 运行所有示例代码
   - 修改参数观察效果
   - 完成所有练习题

2. **深入理解**:
   - 阅读代码注释
   - 理解每个组件的作用
   - 思考优化空间

3. **实验探索**:
   - 尝试不同的配置
   - 对比性能差异
   - 提出改进方案

4. **扩展应用**:
   - 添加新功能
   - 支持更多模型
   - 优化性能

这个教育版的vLLM实现为学习和理解LLM推理系统的核心原理提供了一个优秀的平台。通过简化的实现和丰富的教学材料，学习者可以逐步掌握复杂的推理系统设计思想。