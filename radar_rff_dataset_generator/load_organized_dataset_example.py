#!/usr/bin/env python
"""
双重组织数据集加载示例

演示如何使用重组后的数据集进行不同的研究任务
"""

import h5py
import numpy as np
from pathlib import Path


def example_1_load_by_device():
    """
    示例1: 按设备加载 - 设备指纹识别任务
    
    场景：研究设备5在所有调制方式下的RFF特征
    """
    print("\n" + "="*70)
    print("📱 示例1: 按设备加载数据")
    print("="*70)
    
    device_id = 5
    device_file = f'radar_rff_dataset_organized/by_device/device_{device_id:02d}_Radar_Device_{device_id+1:02d}.h5'
    
    with h5py.File(device_file, 'r') as f:
        # 获取基本信息
        print(f"\n📊 设备信息:")
        print(f"   设备ID: {f.attrs['device_id']}")
        print(f"   设备名称: {f.attrs['device_name']}")
        print(f"   样本总数: {f.attrs['num_samples']:,}")
        print(f"   调制数量: {f.attrs['num_modulations']}")
        print(f"   每种调制样本数: {f.attrs['samples_per_modulation']:,}")
        
        # 加载数据
        iq_data = f['iq_data'][:]
        modulation_labels = f['modulation_labels'][:]
        
        print(f"\n📦 数据形状:")
        print(f"   IQ数据: {iq_data.shape}")
        print(f"   调制标签: {modulation_labels.shape}")
        
        # 统计每种调制的样本数
        print(f"\n📡 各调制方式样本分布:")
        unique_mods, counts = np.unique(modulation_labels, return_counts=True)
        for mod_id, count in zip(unique_mods[:5], counts[:5]):  # 只显示前5个
            print(f"   调制 {mod_id:2d}: {count:,} 样本")
        print(f"   ... (共 {len(unique_mods)} 种调制)")
        
        # 提取特定调制的数据 (例如QPSK, ID=5)
        qpsk_mask = (modulation_labels == 5)
        qpsk_samples = iq_data[qpsk_mask]
        print(f"\n🎯 提取QPSK调制样本:")
        print(f"   样本数: {len(qpsk_samples):,}")
        print(f"   形状: {qpsk_samples.shape}")


def example_2_load_by_modulation():
    """
    示例2: 按调制加载 - 调制识别任务
    
    场景：研究QPSK调制在所有设备上的表现
    """
    print("\n" + "="*70)
    print("📡 示例2: 按调制加载数据")
    print("="*70)
    
    modulation_id = 5  # QPSK
    modulation_name = 'qpsk'
    mod_file = f'radar_rff_dataset_organized/by_modulation/modulation_{modulation_id:02d}_{modulation_name}.h5'
    
    with h5py.File(mod_file, 'r') as f:
        # 获取基本信息
        print(f"\n📊 调制信息:")
        print(f"   调制ID: {f.attrs['modulation_id']}")
        print(f"   调制名称: {f.attrs['modulation_name']}")
        print(f"   样本总数: {f.attrs['num_samples']:,}")
        print(f"   设备数量: {f.attrs['num_devices']}")
        print(f"   每个设备样本数: {f.attrs['samples_per_device']:,}")
        
        # 加载数据
        iq_data = f['iq_data'][:]
        device_labels = f['device_labels'][:]
        
        print(f"\n📦 数据形状:")
        print(f"   IQ数据: {iq_data.shape}")
        print(f"   设备标签: {device_labels.shape}")
        
        # 统计每个设备的样本数
        print(f"\n📱 各设备样本分布:")
        unique_devs, counts = np.unique(device_labels, return_counts=True)
        for dev_id, count in zip(unique_devs[:5], counts[:5]):  # 只显示前5个
            print(f"   设备 {dev_id:2d}: {count:,} 样本")
        print(f"   ... (共 {len(unique_devs)} 个设备)")
        
        # 提取特定设备的数据 (例如设备5)
        device_5_mask = (device_labels == 5)
        device_5_samples = iq_data[device_5_mask]
        print(f"\n🎯 提取设备5的样本:")
        print(f"   样本数: {len(device_5_samples):,}")
        print(f"   形状: {device_5_samples.shape}")


def example_3_batch_loading():
    """
    示例3: 批量加载 - 训练深度学习模型
    
    场景：加载多个设备的数据用于训练
    """
    print("\n" + "="*70)
    print("🔄 示例3: 批量加载多个设备")
    print("="*70)
    
    # 加载设备0-4的所有数据
    device_ids = range(5)
    all_data = []
    all_device_labels = []
    all_mod_labels = []
    
    print("\n📥 正在加载设备...")
    for device_id in device_ids:
        device_file = f'radar_rff_dataset_organized/by_device/device_{device_id:02d}_Radar_Device_{device_id+1:02d}.h5'
        
        with h5py.File(device_file, 'r') as f:
            iq_data = f['iq_data'][:]
            mod_labels = f['modulation_labels'][:]
            
            all_data.append(iq_data)
            all_device_labels.append(np.full(len(iq_data), device_id))
            all_mod_labels.append(mod_labels)
            
            print(f"   ✓ 设备 {device_id}: {len(iq_data):,} 样本")
    
    # 合并数据
    all_data = np.concatenate(all_data, axis=0)
    all_device_labels = np.concatenate(all_device_labels, axis=0)
    all_mod_labels = np.concatenate(all_mod_labels, axis=0)
    
    print(f"\n📊 合并后的数据:")
    print(f"   总样本数: {len(all_data):,}")
    print(f"   数据形状: {all_data.shape}")
    print(f"   设备标签形状: {all_device_labels.shape}")
    print(f"   调制标签形状: {all_mod_labels.shape}")
    
    # 数据集划分示例
    num_samples = len(all_data)
    train_size = int(0.8 * num_samples)
    
    # 打乱数据
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    print(f"\n🎓 数据集划分:")
    print(f"   训练集: {len(train_indices):,} 样本 ({len(train_indices)/num_samples*100:.1f}%)")
    print(f"   测试集: {len(test_indices):,} 样本 ({len(test_indices)/num_samples*100:.1f}%)")


def example_4_cross_modulation_generalization():
    """
    示例4: 跨调制泛化测试
    
    场景：在某些调制上训练，在其他调制上测试
    """
    print("\n" + "="*70)
    print("🔬 示例4: 跨调制泛化测试")
    print("="*70)
    
    device_id = 5
    device_file = f'radar_rff_dataset_organized/by_device/device_{device_id:02d}_Radar_Device_{device_id+1:02d}.h5'
    
    with h5py.File(device_file, 'r') as f:
        iq_data = f['iq_data'][:]
        modulation_labels = f['modulation_labels'][:]
        
        # 训练调制: QAM系列 (0-3)
        train_mods = [0, 1, 2, 3]
        train_mask = np.isin(modulation_labels, train_mods)
        train_data = iq_data[train_mask]
        
        # 测试调制: PSK系列 (4-8)
        test_mods = [4, 5, 6, 7, 8]
        test_mask = np.isin(modulation_labels, test_mods)
        test_data = iq_data[test_mask]
        
        print(f"\n📚 训练集 (QAM系列):")
        print(f"   调制方式: {train_mods}")
        print(f"   样本数: {len(train_data):,}")
        
        print(f"\n🧪 测试集 (PSK系列):")
        print(f"   调制方式: {test_mods}")
        print(f"   样本数: {len(test_data):,}")
        
        print(f"\n💡 用途: 测试模型在未见过的调制方式上的泛化能力")


def example_5_pytorch_dataloader():
    """
    示例5: PyTorch DataLoader集成
    
    场景：创建PyTorch数据加载器用于训练
    """
    print("\n" + "="*70)
    print("🔥 示例5: PyTorch DataLoader集成")
    print("="*70)
    
    print("\n📝 PyTorch Dataset类示例代码:")
    print("""
from torch.utils.data import Dataset, DataLoader
import h5py
import torch

class RadarRFFDataset(Dataset):
    def __init__(self, device_ids, base_dir='radar_rff_dataset_organized/by_device'):
        self.data = []
        self.device_labels = []
        self.mod_labels = []
        
        # 加载指定设备的数据
        for device_id in device_ids:
            filename = f'{base_dir}/device_{device_id:02d}_Radar_Device_{device_id+1:02d}.h5'
            with h5py.File(filename, 'r') as f:
                self.data.append(f['iq_data'][:])
                self.mod_labels.append(f['modulation_labels'][:])
                self.device_labels.append(np.full(len(f['iq_data']), device_id))
        
        self.data = np.concatenate(self.data, axis=0)
        self.device_labels = np.concatenate(self.device_labels, axis=0)
        self.mod_labels = np.concatenate(self.mod_labels, axis=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回IQ数据和标签
        iq_sample = torch.from_numpy(self.data[idx]).float()
        device_label = torch.tensor(self.device_labels[idx]).long()
        mod_label = torch.tensor(self.mod_labels[idx]).long()
        
        return iq_sample, device_label, mod_label

# 使用示例
train_dataset = RadarRFFDataset(device_ids=[0, 1, 2, 3, 4])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# 训练循环
for batch_idx, (iq_data, device_labels, mod_labels) in enumerate(train_loader):
    # iq_data shape: [64, 2, 2049]
    # device_labels shape: [64]
    # mod_labels shape: [64]
    
    # 你的训练代码...
    pass
""")


def example_6_dataset_statistics():
    """
    示例6: 数据集统计分析
    
    场景：分析数据集的整体特征
    """
    print("\n" + "="*70)
    print("📈 示例6: 数据集统计分析")
    print("="*70)
    
    # 随机选择一个设备文件进行分析
    device_id = 0
    device_file = f'radar_rff_dataset_organized/by_device/device_{device_id:02d}_Radar_Device_{device_id+1:02d}.h5'
    
    with h5py.File(device_file, 'r') as f:
        # 读取少量样本进行统计 (避免内存溢出)
        sample_size = 1000
        iq_data = f['iq_data'][:sample_size]
        
        # 计算统计量
        i_channel = iq_data[:, 0, :]  # I通道
        q_channel = iq_data[:, 1, :]  # Q通道
        
        print(f"\n📊 信号统计 (基于{sample_size}个样本):")
        print(f"\n   I通道:")
        print(f"      均值: {i_channel.mean():.6f}")
        print(f"      标准差: {i_channel.std():.6f}")
        print(f"      最小值: {i_channel.min():.6f}")
        print(f"      最大值: {i_channel.max():.6f}")
        
        print(f"\n   Q通道:")
        print(f"      均值: {q_channel.mean():.6f}")
        print(f"      标准差: {q_channel.std():.6f}")
        print(f"      最小值: {q_channel.min():.6f}")
        print(f"      最大值: {q_channel.max():.6f}")
        
        # 计算信号功率
        power = i_channel**2 + q_channel**2
        print(f"\n   信号功率:")
        print(f"      平均功率: {power.mean():.6f}")
        print(f"      功率标准差: {power.std():.6f}")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("🎯 双重组织数据集加载示例")
    print("="*70)
    print("\n本脚本演示如何使用重组后的数据集")
    print("请确保已经运行过 organize_dataset.py 生成了重组数据")
    
    # 检查数据集是否存在
    base_dir = Path('radar_rff_dataset_organized')
    if not base_dir.exists():
        print(f"\n❌ 错误: 找不到重组后的数据集目录: {base_dir}")
        print("   请先运行: python organize_dataset.py")
        return
    
    try:
        # 运行示例 (注释掉不需要的示例)
        example_1_load_by_device()
        example_2_load_by_modulation()
        example_3_batch_loading()
        example_4_cross_modulation_generalization()
        example_5_pytorch_dataloader()
        example_6_dataset_statistics()
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("   请确保数据集已经生成并重组")
    
    print("\n" + "="*70)
    print("✅ 所有示例运行完成！")
    print("="*70)


if __name__ == '__main__':
    main()

