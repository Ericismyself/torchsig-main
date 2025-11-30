#!/usr/bin/env python
"""
数据集可视化工具

用于可视化生成的雷达RFF数据集，包括：
- 类别分布统计
- 样本可视化（星座图、频谱等）
- RFF特征分析

作者：根据研究需求生成
日期：2025-10-18
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import yaml
from collections import Counter

def load_dataset(dataset_path):
    """加载数据集"""
    print(f"加载数据集: {dataset_path}")
    
    if dataset_path.endswith('.h5'):
        # HDF5格式
        f = h5py.File(dataset_path, 'r')
        data = {
            'iq_data': f['iq_data'],
            'device_labels': f['device_labels'][:],
            'modulation_labels': f['modulation_labels'][:],
            'combined_labels': f['combined_labels'][:],
            'snr_db': f['snr_db'][:],
        }
        return data, f
    else:
        # NumPy格式（假设在同一目录）
        import os
        base_dir = dataset_path if os.path.isdir(dataset_path) else os.path.dirname(dataset_path)
        
        data = {
            'iq_data': np.load(os.path.join(base_dir, 'iq_data.npy'), mmap_mode='r'),
            'device_labels': np.load(os.path.join(base_dir, 'device_labels.npy')),
            'modulation_labels': np.load(os.path.join(base_dir, 'modulation_labels.npy')),
            'combined_labels': np.load(os.path.join(base_dir, 'combined_labels.npy')),
            'snr_db': np.load(os.path.join(base_dir, 'snr_db.npy')),
        }
        return data, None


def plot_class_distribution(data, metadata_path, save_path='class_distribution.png'):
    """绘制类别分布"""
    print("\n绘制类别分布...")
    
    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)
    
    device_names = metadata['device_list']
    mod_names = metadata['modulation_list']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. 设备标签分布
    ax1 = fig.add_subplot(gs[0, 0])
    device_counts = Counter(data['device_labels'])
    devices = sorted(device_counts.keys())
    counts = [device_counts[d] for d in devices]
    ax1.bar(devices, counts)
    ax1.set_xlabel('设备 ID')
    ax1.set_ylabel('样本数')
    ax1.set_title(f'设备标签分布 ({len(devices)} 个设备)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 调制标签分布
    ax2 = fig.add_subplot(gs[0, 1])
    mod_counts = Counter(data['modulation_labels'])
    mods = sorted(mod_counts.keys())
    counts = [mod_counts[m] for m in mods]
    ax2.bar(mods, counts)
    ax2.set_xlabel('调制 ID')
    ax2.set_ylabel('样本数')
    ax2.set_title(f'调制标签分布 ({len(mods)} 种调制)')
    ax2.set_xticks(range(0, len(mods), 2))
    ax2.grid(True, alpha=0.3)
    
    # 3. SNR分布
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(data['snr_db'], bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('样本数')
    ax3.set_title(f'SNR分布 (均值={np.mean(data["snr_db"]):.2f} dB)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 组合类别分布（热力图）
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 创建设备-调制矩阵
    num_devices = len(device_names)
    num_mods = len(mod_names)
    matrix = np.zeros((num_devices, num_mods))
    
    for dev_label, mod_label in zip(data['device_labels'], data['modulation_labels']):
        matrix[dev_label, mod_label] += 1
    
    im = ax4.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax4.set_xlabel('调制 ID')
    ax4.set_ylabel('设备 ID')
    ax4.set_title('设备-调制样本数热力图')
    plt.colorbar(im, ax=ax4, label='样本数')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")


def plot_sample_visualization(data, metadata_path, num_samples=6, save_path='sample_visualization.png'):
    """可视化随机样本"""
    print("\n可视化随机样本...")
    
    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)
    
    device_names = metadata['device_list']
    mod_names = metadata['modulation_list']
    sample_rate = metadata['signal_parameters']['sample_rate_hz']
    
    # 随机选择样本
    total_samples = len(data['device_labels'])
    indices = np.random.choice(total_samples, num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3*num_samples))
    
    for i, idx in enumerate(indices):
        # 读取数据
        iq_signal = data['iq_data'][idx]
        device_id = data['device_labels'][idx]
        mod_id = data['modulation_labels'][idx]
        snr = data['snr_db'][idx]
        
        device_name = device_names[device_id]
        mod_name = mod_names[mod_id]
        
        # 星座图
        ax = axes[i, 0]
        ax.scatter(np.real(iq_signal[::10]), np.imag(iq_signal[::10]), 
                  s=1, alpha=0.3)
        ax.set_title(f'{device_name} - {mod_name}\nSNR={snr:.1f}dB')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        
        # 时域幅度
        ax = axes[i, 1]
        ax.plot(np.abs(iq_signal[:500]), linewidth=0.5)
        ax.set_title('时域幅度')
        ax.set_xlabel('样本')
        ax.set_ylabel('|IQ|')
        ax.grid(True, alpha=0.3)
        
        # 时域相位
        ax = axes[i, 2]
        ax.plot(np.angle(iq_signal[:500]), linewidth=0.5)
        ax.set_title('时域相位')
        ax.set_xlabel('样本')
        ax.set_ylabel('相位 (rad)')
        ax.grid(True, alpha=0.3)
        
        # 功率谱
        ax = axes[i, 3]
        fft = np.fft.fftshift(np.fft.fft(iq_signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_signal), 1/sample_rate))
        ax.plot(freqs/1e3, 20*np.log10(np.abs(fft) + 1e-10), linewidth=0.5)
        ax.set_title('功率谱')
        ax.set_xlabel('频率 (kHz)')
        ax.set_ylabel('功率 (dB)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")


def plot_rff_comparison(data, metadata_path, mod_id=0, num_devices=5, save_path='rff_comparison.png'):
    """对比不同设备的同一调制信号"""
    print(f"\n对比不同设备的RFF特征...")
    
    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)
    
    device_names = metadata['device_list']
    mod_names = metadata['modulation_list']
    mod_name = mod_names[mod_id]
    sample_rate = metadata['signal_parameters']['sample_rate_hz']
    
    # 为每个设备选择一个样本（同一调制，相近SNR）
    device_signals = {}
    target_snr = 20  # 目标SNR
    
    for device_id in range(min(num_devices, len(device_names))):
        # 找到该设备-调制组合的样本
        mask = (data['device_labels'] == device_id) & (data['modulation_labels'] == mod_id)
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            # 选择SNR最接近目标的样本
            snrs = data['snr_db'][indices]
            best_idx = indices[np.argmin(np.abs(snrs - target_snr))]
            
            device_signals[device_id] = {
                'signal': data['iq_data'][best_idx],
                'snr': data['snr_db'][best_idx],
                'name': device_names[device_id],
            }
    
    # 可视化
    num_devs = len(device_signals)
    fig, axes = plt.subplots(num_devs, 3, figsize=(15, 3*num_devs))
    
    if num_devs == 1:
        axes = axes.reshape(1, -1)
    
    for i, (device_id, info) in enumerate(device_signals.items()):
        signal = info['signal']
        device_name = info['name']
        snr = info['snr']
        
        # 星座图
        ax = axes[i, 0]
        ax.scatter(np.real(signal[::10]), np.imag(signal[::10]), 
                  s=2, alpha=0.3)
        ax.set_title(f'{device_name}\n星座图')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 时域波形
        ax = axes[i, 1]
        ax.plot(np.real(signal[:500]), alpha=0.7, label='I', linewidth=0.5)
        ax.plot(np.imag(signal[:500]), alpha=0.7, label='Q', linewidth=0.5)
        ax.set_title(f'时域波形 (SNR={snr:.1f}dB)')
        ax.set_xlabel('样本')
        ax.set_ylabel('幅度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 功率谱
        ax = axes[i, 2]
        fft = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/sample_rate))
        ax.plot(freqs/1e3, 20*np.log10(np.abs(fft) + 1e-10), linewidth=0.5)
        ax.set_title('功率谱')
        ax.set_xlabel('频率 (kHz)')
        ax.set_ylabel('功率 (dB)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'不同设备的RFF特征对比 - {mod_name}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")


def print_dataset_summary(data, metadata_path):
    """打印数据集摘要"""
    print("\n" + "="*80)
    print("数据集摘要")
    print("="*80)
    
    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)
    
    print(f"\n基本信息:")
    print(f"  - 数据集名称: {metadata['dataset_info']['name']}")
    print(f"  - 创建日期: {metadata['dataset_info']['creation_date']}")
    print(f"  - 设备数量: {metadata['dataset_info']['num_devices']}")
    print(f"  - 调制数量: {metadata['dataset_info']['num_modulations']}")
    print(f"  - 总类别数: {metadata['dataset_info']['total_classes']}")
    print(f"  - 总样本数: {metadata['dataset_info']['total_samples']}")
    
    print(f"\n信号参数:")
    print(f"  - IQ采样点数: {metadata['signal_parameters']['num_iq_samples']}")
    print(f"  - 采样率: {metadata['signal_parameters']['sample_rate_hz']/1e6} MHz")
    print(f"  - SNR范围: {metadata['signal_parameters']['snr_range_db']} dB")
    
    print(f"\n实际统计:")
    print(f"  - 样本数: {len(data['device_labels'])}")
    print(f"  - 设备类别: {len(np.unique(data['device_labels']))}")
    print(f"  - 调制类别: {len(np.unique(data['modulation_labels']))}")
    print(f"  - 组合类别: {len(np.unique(data['combined_labels']))}")
    print(f"  - SNR范围: [{np.min(data['snr_db']):.2f}, {np.max(data['snr_db']):.2f}] dB")
    print(f"  - SNR平均值: {np.mean(data['snr_db']):.2f} dB")
    
    print(f"\n数据大小:")
    iq_size_gb = data['iq_data'].shape[0] * data['iq_data'].shape[1] * 8 / (1024**3)
    print(f"  - IQ数据: {iq_size_gb:.2f} GB")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='雷达RFF数据集可视化工具')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='数据集路径 (HDF5文件或NumPy目录)')
    parser.add_argument('--metadata', type=str, required=True,
                       help='元数据文件路径 (metadata.yaml)')
    parser.add_argument('--output-dir', type=str, default='./',
                       help='输出图片保存目录')
    parser.add_argument('--num-samples', type=int, default=6,
                       help='可视化样本数量')
    parser.add_argument('--num-devices', type=int, default=5,
                       help='RFF对比中的设备数量')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    data, h5_file = load_dataset(args.dataset)
    
    try:
        # 打印摘要
        print_dataset_summary(data, args.metadata)
        
        # 绘制类别分布
        plot_class_distribution(
            data, args.metadata,
            save_path=os.path.join(args.output_dir, 'class_distribution.png')
        )
        
        # 可视化样本
        plot_sample_visualization(
            data, args.metadata,
            num_samples=args.num_samples,
            save_path=os.path.join(args.output_dir, 'sample_visualization.png')
        )
        
        # RFF特征对比
        plot_rff_comparison(
            data, args.metadata,
            mod_id=0,  # 使用第一种调制
            num_devices=args.num_devices,
            save_path=os.path.join(args.output_dir, 'rff_comparison.png')
        )
        
        print("\n✅ 可视化完成！")
        print(f"图片已保存至: {args.output_dir}/")
        
    finally:
        if h5_file is not None:
            h5_file.close()


if __name__ == "__main__":
    main()

