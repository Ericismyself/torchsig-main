#!/usr/bin/env python
"""
快速检查数据集标签的脚本

运行方式：
    python check_labels.py
    python check_labels.py --dataset-path ./radar_rff_dataset_realistic/radar_rff_dataset.h5
"""
import h5py
import numpy as np
import argparse
import os

def check_labels(dataset_path):
    """检查数据集标签"""
    
    if not os.path.exists(dataset_path):
        print(f"❌ 找不到数据集文件: {dataset_path}")
        print(f"   请先生成数据集")
        return
    
    print("\n" + "="*80)
    print("🔍 数据集标签检查")
    print("="*80)
    print(f"文件: {dataset_path}")
    
    # 打开数据集
    with h5py.File(dataset_path, 'r') as f:
        # 显示所有数据集
        print("\n📂 HDF5文件中的数据集:")
        for key in f.keys():
            shape = f[key].shape
            dtype = f[key].dtype
            print(f"   {key:<25} shape={shape}, dtype={dtype}")
        
        # 读取标签
        iq_data = f['iq_data']
        device_labels = f['device_labels'][:]
        modulation_labels = f['modulation_labels'][:]
        combined_labels = f['combined_labels'][:]
        snr_db = f['snr_db'][:]
        
        total_samples = len(device_labels)
        
        print("\n" + "="*80)
        print("📊 标签统计信息")
        print("="*80)
        
        # 基本信息
        print(f"\n总样本数: {total_samples:,}")
        print(f"IQ数据shape: {iq_data.shape}")
        print(f"IQ数据dtype: {iq_data.dtype}")
        
        # 设备标签
        print(f"\n📱 设备标签 (device_labels):")
        print(f"   范围: {device_labels.min()} ~ {device_labels.max()}")
        print(f"   唯一设备数: {len(np.unique(device_labels))} 个")
        device_counts = np.bincount(device_labels)
        print(f"   每个设备的样本数:")
        for dev_id, count in enumerate(device_counts):
            print(f"      设备 {dev_id:2d} (Radar_Device_{dev_id+1:02d}): {count:7,} 样本")
        
        # 调制标签
        print(f"\n📡 调制标签 (modulation_labels):")
        print(f"   范围: {modulation_labels.min()} ~ {modulation_labels.max()}")
        print(f"   唯一调制数: {len(np.unique(modulation_labels))} 种")
        modulation_counts = np.bincount(modulation_labels)
        
        # 26种调制的名称（按照配置顺序）
        modulation_names = [
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
        
        print(f"   每种调制的样本数:")
        for mod_id, count in enumerate(modulation_counts):
            mod_name = modulation_names[mod_id] if mod_id < len(modulation_names) else 'unknown'
            print(f"      调制 {mod_id:2d} ({mod_name:<15}): {count:7,} 样本")
        
        # 组合标签
        print(f"\n🔗 组合标签 (combined_labels):")
        print(f"   范围: {combined_labels.min()} ~ {combined_labels.max()}")
        print(f"   唯一类别数: {len(np.unique(combined_labels))} 类")
        combined_counts = np.bincount(combined_labels)
        samples_per_class = combined_counts[0]
        print(f"   每类样本数: {samples_per_class:,} (应该全部相同)")
        
        # 检查是否所有类别样本数相同
        if np.all(combined_counts == samples_per_class):
            print(f"   ✅ 所有类别样本数均为 {samples_per_class}")
        else:
            print(f"   ⚠️  类别样本数不一致！")
            unique_counts = np.unique(combined_counts)
            print(f"   样本数分布: {unique_counts}")
        
        # SNR信息
        print(f"\n📶 SNR信息 (snr_db):")
        print(f"   范围: {snr_db.min():.2f} ~ {snr_db.max():.2f} dB")
        print(f"   平均值: {snr_db.mean():.2f} dB")
        print(f"   中位数: {np.median(snr_db):.2f} dB")
        print(f"   标准差: {snr_db.std():.2f} dB")
        
        # 显示前10个样本
        print(f"\n📋 前10个样本的标签示例:")
        print(f"   {'索引':<8} {'设备ID':<10} {'调制ID':<10} {'组合ID':<10} {'SNR(dB)':<10}")
        print("   " + "-"*60)
        for i in range(min(10, total_samples)):
            print(f"   {i:<8} {device_labels[i]:<10} {modulation_labels[i]:<10} "
                  f"{combined_labels[i]:<10} {snr_db[i]:<10.2f}")
        
        # 验证组合ID计算
        print(f"\n🔍 验证组合ID计算公式:")
        print(f"   公式: combined_id = device_id * 26 + modulation_id")
        
        calculated_combined = device_labels * 26 + modulation_labels
        if np.all(calculated_combined == combined_labels):
            print(f"   ✅ 所有组合ID计算正确")
        else:
            print(f"   ❌ 组合ID计算有误！")
            mismatch_count = np.sum(calculated_combined != combined_labels)
            print(f"   不匹配的样本数: {mismatch_count}")
        
        # 数据完整性检查
        print(f"\n✅ 数据完整性检查:")
        print(f"   设备ID范围正确: {0 <= device_labels.min() and device_labels.max() < 20}")
        print(f"   调制ID范围正确: {0 <= modulation_labels.min() and modulation_labels.max() < 26}")
        print(f"   组合ID范围正确: {0 <= combined_labels.min() and combined_labels.max() < 520}")
        print(f"   所有标签长度一致: {len(device_labels) == len(modulation_labels) == len(combined_labels)}")
        
        # 数据集质量评估
        print(f"\n📈 数据集质量评估:")
        num_devices = len(np.unique(device_labels))
        num_modulations = len(np.unique(modulation_labels))
        num_classes = len(np.unique(combined_labels))
        expected_classes = num_devices * num_modulations
        
        print(f"   设备数: {num_devices} (期望: 20)")
        print(f"   调制数: {num_modulations} (期望: 26)")
        print(f"   总类别数: {num_classes} (期望: {expected_classes})")
        print(f"   每类样本数: {samples_per_class:,}")
        
        if num_classes == expected_classes and samples_per_class > 0:
            print(f"\n   🎉 数据集完整且符合预期！")
        else:
            print(f"\n   ⚠️  数据集可能不完整，请检查")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='检查雷达RFF数据集标签')
    parser.add_argument('--dataset-path', 
                       default='./radar_rff_dataset_realistic/radar_rff_dataset.h5',
                       help='数据集HDF5文件路径')
    
    args = parser.parse_args()
    check_labels(args.dataset_path)

if __name__ == '__main__':
    main()

