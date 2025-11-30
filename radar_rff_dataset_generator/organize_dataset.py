#!/usr/bin/env python
"""
数据集重组工具 - 双重组织结构

将单个大HDF5文件重组为两种结构：
1. 按设备组织 (20个文件): 每个文件包含1个设备的26种调制
2. 按调制组织 (26个文件): 每个文件包含1种调制的20个设备

使用示例：
    python organize_dataset.py \
        --input-h5 radar_rff_dataset_realistic/radar_rff_dataset.h5 \
        --output-dir radar_rff_dataset_organized \
        --num-workers 4
"""

import h5py
import numpy as np
import argparse
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil


# 26种调制方式的名称（按照ID顺序）
MODULATION_NAMES = [
    # QAM系列 (4种)
    '16qam', '64qam', '256qam', '1024qam',
    # PSK系列 (5种)
    'bpsk', 'qpsk', '8psk', '16psk', '32psk',
    # FSK系列 (4种)
    '2fsk', '4fsk', '8fsk', '16fsk',
    # GFSK系列 (4种)
    '2gfsk', '4gfsk', '8gfsk', '16gfsk',
    # MSK系列 (4种)
    '2msk', '4msk', '8msk', '16msk',
    # AM系列 (4种)
    'am-dsb', 'am-dsb-sc', 'am-lsb', 'am-usb',
    # FM系列 (1种)
    'fm'
]


def organize_by_device(input_h5_path, output_dir, num_devices=20, num_modulations=26):
    """
    按设备组织数据集
    每个文件包含1个设备的所有26种调制
    """
    print("\n" + "="*70)
    print("📱 按设备组织数据集")
    print("="*70)
    
    # 创建输出目录
    device_dir = Path(output_dir) / "by_device"
    device_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开输入文件
    with h5py.File(input_h5_path, 'r') as f_in:
        iq_data = f_in['iq_data']
        device_labels = f_in['device_labels'][:]
        modulation_labels = f_in['modulation_labels'][:]
        combined_labels = f_in['combined_labels'][:]
        
        # 获取数据形状
        total_samples, num_channels, num_iq_samples = iq_data.shape
        samples_per_class = total_samples // (num_devices * num_modulations)
        
        print(f"📊 数据集信息:")
        print(f"   总样本数: {total_samples:,}")
        print(f"   每类样本数: {samples_per_class:,}")
        print(f"   IQ采样点数: {num_iq_samples}")
        print(f"\n🔄 开始按设备组织...")
        
        # 为每个设备创建一个HDF5文件
        for device_id in tqdm(range(num_devices), desc="处理设备"):
            # 找到该设备的所有样本索引
            device_mask = (device_labels == device_id)
            device_indices = np.where(device_mask)[0]
            
            # 输出文件名
            device_name = f"Radar_Device_{device_id+1:02d}"
            output_file = device_dir / f"device_{device_id:02d}_{device_name}.h5"
            
            # 创建输出文件
            with h5py.File(output_file, 'w') as f_out:
                # 读取该设备的所有数据
                device_iq_data = iq_data[device_indices]
                device_mod_labels = modulation_labels[device_indices]
                device_combined_labels = combined_labels[device_indices]
                
                # 写入数据
                f_out.create_dataset('iq_data', data=device_iq_data, 
                                    compression='gzip', compression_opts=4)
                f_out.create_dataset('modulation_labels', data=device_mod_labels, 
                                    compression='gzip')
                f_out.create_dataset('combined_labels', data=device_combined_labels, 
                                    compression='gzip')
                
                # 添加属性
                f_out.attrs['device_id'] = device_id
                f_out.attrs['device_name'] = device_name
                f_out.attrs['num_samples'] = len(device_indices)
                f_out.attrs['num_modulations'] = num_modulations
                f_out.attrs['samples_per_modulation'] = samples_per_class
                f_out.attrs['num_iq_samples'] = num_iq_samples
    
    print(f"✅ 按设备组织完成！共生成 {num_devices} 个文件")
    print(f"   输出目录: {device_dir}")


def organize_by_modulation(input_h5_path, output_dir, num_devices=20, num_modulations=26):
    """
    按调制组织数据集
    每个文件包含1种调制的所有20个设备
    """
    print("\n" + "="*70)
    print("📡 按调制组织数据集")
    print("="*70)
    
    # 创建输出目录
    modulation_dir = Path(output_dir) / "by_modulation"
    modulation_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开输入文件
    with h5py.File(input_h5_path, 'r') as f_in:
        iq_data = f_in['iq_data']
        device_labels = f_in['device_labels'][:]
        modulation_labels = f_in['modulation_labels'][:]
        combined_labels = f_in['combined_labels'][:]
        
        # 获取数据形状
        total_samples, num_channels, num_iq_samples = iq_data.shape
        samples_per_class = total_samples // (num_devices * num_modulations)
        
        print(f"📊 数据集信息:")
        print(f"   总样本数: {total_samples:,}")
        print(f"   每类样本数: {samples_per_class:,}")
        print(f"   IQ采样点数: {num_iq_samples}")
        print(f"\n🔄 开始按调制组织...")
        
        # 为每种调制创建一个HDF5文件
        for mod_id in tqdm(range(num_modulations), desc="处理调制"):
            # 找到该调制的所有样本索引
            mod_mask = (modulation_labels == mod_id)
            mod_indices = np.where(mod_mask)[0]
            
            # 输出文件名
            mod_name = MODULATION_NAMES[mod_id]
            output_file = modulation_dir / f"modulation_{mod_id:02d}_{mod_name}.h5"
            
            # 创建输出文件
            with h5py.File(output_file, 'w') as f_out:
                # 读取该调制的所有数据
                mod_iq_data = iq_data[mod_indices]
                mod_device_labels = device_labels[mod_indices]
                mod_combined_labels = combined_labels[mod_indices]
                
                # 写入数据
                f_out.create_dataset('iq_data', data=mod_iq_data, 
                                    compression='gzip', compression_opts=4)
                f_out.create_dataset('device_labels', data=mod_device_labels, 
                                    compression='gzip')
                f_out.create_dataset('combined_labels', data=mod_combined_labels, 
                                    compression='gzip')
                
                # 添加属性
                f_out.attrs['modulation_id'] = mod_id
                f_out.attrs['modulation_name'] = mod_name
                f_out.attrs['num_samples'] = len(mod_indices)
                f_out.attrs['num_devices'] = num_devices
                f_out.attrs['samples_per_device'] = samples_per_class
                f_out.attrs['num_iq_samples'] = num_iq_samples
    
    print(f"✅ 按调制组织完成！共生成 {num_modulations} 个文件")
    print(f"   输出目录: {modulation_dir}")


def copy_metadata_files(input_dir, output_dir):
    """复制元数据文件"""
    print("\n" + "="*70)
    print("📋 复制元数据文件")
    print("="*70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 需要复制的文件
    metadata_files = [
        'metadata.yaml',
        'device_rff_parameters.yaml',
        'class_mapping.yaml'
    ]
    
    for filename in metadata_files:
        src = input_path / filename
        dst = output_path / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"   ✓ {filename}")
        else:
            print(f"   ⚠ {filename} (未找到)")
    
    print("✅ 元数据文件复制完成")


def create_readme(output_dir, num_devices=20, num_modulations=26, samples_per_class=3000):
    """创建README文件"""
    readme_content = f"""# 雷达RFF数据集 - 双重组织结构

## 📁 目录结构

```
{Path(output_dir).name}/
├── by_device/              # 按设备组织 ({num_devices}个文件)
│   ├── device_00_Radar_Device_01.h5
│   ├── device_01_Radar_Device_02.h5
│   └── ...
│
├── by_modulation/          # 按调制组织 ({num_modulations}个文件)
│   ├── modulation_00_16qam.h5
│   ├── modulation_01_64qam.h5
│   └── ...
│
├── metadata.yaml
├── device_rff_parameters.yaml
└── class_mapping.yaml
```

## 📊 数据集规模

- **设备数量**: {num_devices} 类
- **调制方式**: {num_modulations} 种
- **总类别数**: {num_devices} × {num_modulations} = {num_devices * num_modulations} 类
- **每类样本数**: {samples_per_class:,}
- **总样本数**: {num_devices * num_modulations * samples_per_class:,}

## 🎯 使用场景

### 1️⃣ 按设备组织 (`by_device/`)

**适用于**: 设备指纹识别、设备分类任务

每个文件包含**1个设备**的**{num_modulations}种调制**:
- 样本数: {num_modulations * samples_per_class:,} 个
- 数据形状: [{num_modulations * samples_per_class:,}, 2, 2049]
- 标签: modulation_labels (0-{num_modulations-1})

```python
import h5py

# 加载设备5的所有数据
with h5py.File('by_device/device_05_Radar_Device_06.h5', 'r') as f:
    iq_data = f['iq_data'][:]              # [{num_modulations * samples_per_class:,}, 2, 2049]
    mod_labels = f['modulation_labels'][:]  # [{num_modulations * samples_per_class:,}] (0-{num_modulations-1})
    
    print(f"设备ID: {{f.attrs['device_id']}}")
    print(f"设备名称: {{f.attrs['device_name']}}")
```

### 2️⃣ 按调制组织 (`by_modulation/`)

**适用于**: 调制识别、调制分类任务

每个文件包含**1种调制**的**{num_devices}个设备**:
- 样本数: {num_devices * samples_per_class:,} 个
- 数据形状: [{num_devices * samples_per_class:,}, 2, 2049]
- 标签: device_labels (0-{num_devices-1})

```python
import h5py

# 加载QPSK调制的所有数据
with h5py.File('by_modulation/modulation_05_qpsk.h5', 'r') as f:
    iq_data = f['iq_data'][:]               # [{num_devices * samples_per_class:,}, 2, 2049]
    dev_labels = f['device_labels'][:]      # [{num_devices * samples_per_class:,}] (0-{num_devices-1})
    
    print(f"调制ID: {{f.attrs['modulation_id']}}")
    print(f"调制名称: {{f.attrs['modulation_name']}}")
```

## 🔍 数据格式

### HDF5数据集结构

**by_device/<device>.h5**:
- `iq_data`: [N, 2, 2049] - IQ采样数据
- `modulation_labels`: [N] - 调制标签 (0-{num_modulations-1})
- `combined_labels`: [N] - 组合标签 (用于全局索引)

**by_modulation/<modulation>.h5**:
- `iq_data`: [N, 2, 2049] - IQ采样数据
- `device_labels`: [N] - 设备标签 (0-{num_devices-1})
- `combined_labels`: [N] - 组合标签 (用于全局索引)

## 📝 调制方式列表

| ID | 调制名称 | 类型 |
|----|---------|------|
| 0-3 | 16qam, 64qam, 256qam, 1024qam | QAM |
| 4-8 | bpsk, qpsk, 8psk, 16psk, 32psk | PSK |
| 9-12 | 2fsk, 4fsk, 8fsk, 16fsk | FSK |
| 13-16 | 2gfsk, 4gfsk, 8gfsk, 16gfsk | GFSK |
| 17-20 | 2msk, 4msk, 8msk, 16msk | MSK |
| 21-24 | am-dsb, am-dsb-sc, am-lsb, am-usb | AM |
| 25 | fm | FM |

## 💡 使用示例

### 示例1: 加载特定设备的所有调制数据

```python
import h5py
import numpy as np

device_id = 5
with h5py.File(f'by_device/device_{{device_id:02d}}_Radar_Device_{{device_id+1:02d}}.h5', 'r') as f:
    # 获取所有QPSK调制的样本 (调制ID=5)
    qpsk_mask = (f['modulation_labels'][:] == 5)
    qpsk_samples = f['iq_data'][qpsk_mask]
    print(f"设备{{device_id}}的QPSK样本数: {{len(qpsk_samples)}}")
```

### 示例2: 加载特定调制的所有设备数据

```python
import h5py

with h5py.File('by_modulation/modulation_05_qpsk.h5', 'r') as f:
    # 获取设备5的样本 (设备ID=5)
    device_mask = (f['device_labels'][:] == 5)
    device_samples = f['iq_data'][device_mask]
    print(f"QPSK调制下设备5的样本数: {{len(device_samples)}}")
```

### 示例3: 批量加载多个设备

```python
import h5py
import numpy as np

# 加载设备0-4的所有数据
all_data = []
all_labels = []

for device_id in range(5):
    filename = f'by_device/device_{{device_id:02d}}_Radar_Device_{{device_id+1:02d}}.h5'
    with h5py.File(filename, 'r') as f:
        all_data.append(f['iq_data'][:])
        all_labels.append(np.full(len(f['iq_data']), device_id))

all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
print(f"总样本数: {{len(all_data)}}")
```

## 🎓 适用任务

1. **设备指纹识别** (RFF): 使用 `by_device/` 或 `by_modulation/`
2. **调制识别**: 使用 `by_device/`
3. **联合分类** (设备+调制): 使用任一组织方式
4. **跨调制泛化**: 使用 `by_device/` 进行训练/测试分离
5. **跨设备泛化**: 使用 `by_modulation/` 进行训练/测试分离

---

生成时间: {np.datetime64('now')}
"""
    
    readme_path = Path(output_dir) / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ README文件已创建: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将单个HDF5数据集重组为双重结构（按设备和按调制）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-h5', type=str, required=True,
                       help='输入HDF5文件路径')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='输入目录（包含元数据文件），默认为input-h5所在目录')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num-devices', type=int, default=20,
                       help='设备数量 (默认: 20)')
    parser.add_argument('--num-modulations', type=int, default=26,
                       help='调制数量 (默认: 26)')
    parser.add_argument('--organize-by-device', action='store_true', default=True,
                       help='按设备组织 (默认: True)')
    parser.add_argument('--organize-by-modulation', action='store_true', default=True,
                       help='按调制组织 (默认: True)')
    parser.add_argument('--skip-device', action='store_true',
                       help='跳过按设备组织')
    parser.add_argument('--skip-modulation', action='store_true',
                       help='跳过按调制组织')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_h5):
        print(f"❌ 错误: 输入文件不存在: {args.input_h5}")
        return
    
    # 确定输入目录
    if args.input_dir is None:
        args.input_dir = os.path.dirname(args.input_h5)
    
    print("\n" + "="*70)
    print("🔄 数据集重组工具")
    print("="*70)
    print(f"📂 输入文件: {args.input_h5}")
    print(f"📂 输入目录: {args.input_dir}")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"📱 设备数量: {args.num_devices}")
    print(f"📡 调制数量: {args.num_modulations}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 按设备组织
    if not args.skip_device:
        organize_by_device(args.input_h5, args.output_dir, 
                          args.num_devices, args.num_modulations)
    
    # 按调制组织
    if not args.skip_modulation:
        organize_by_modulation(args.input_h5, args.output_dir,
                              args.num_devices, args.num_modulations)
    
    # 复制元数据文件
    copy_metadata_files(args.input_dir, args.output_dir)
    
    # 创建README
    create_readme(args.output_dir, args.num_devices, args.num_modulations)
    
    print("\n" + "="*70)
    print("🎉 数据集重组完成！")
    print("="*70)
    print(f"📁 输出目录: {args.output_dir}")
    print(f"   ├── by_device/       ({args.num_devices} 个文件)")
    print(f"   ├── by_modulation/   ({args.num_modulations} 个文件)")
    print(f"   ├── metadata.yaml")
    print(f"   ├── README.md")
    print(f"   └── ...")


if __name__ == '__main__':
    main()

