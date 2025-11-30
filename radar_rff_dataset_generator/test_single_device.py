#!/usr/bin/env python
"""
测试单个设备的信号生成

快速验证配置是否正确，生成少量样本用于测试和可视化

作者：根据研究需求生成
日期：2025-10-18
"""

import numpy as np
import matplotlib.pyplot as plt
from config_rff_parameters import generate_device_parameters, MODULATION_LIST
from rff_impairments import RFFImpairmentSimulator, add_awgn, normalize_signal_power
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.datasets.dataset_metadata import DatasetMetadata

print("="*80)
print("单设备测试脚本")
print("="*80)

# 配置
NUM_TEST_DEVICES = 3
NUM_TEST_MODULATIONS = 3
SAMPLE_RATE = 1e6
NUM_IQ_SAMPLES = 4096
SNR_DB = 20

# 生成测试设备
print(f"\n生成 {NUM_TEST_DEVICES} 个测试设备...")
devices = generate_device_parameters(num_devices=NUM_TEST_DEVICES, seed=42, diverse=True)

for device in devices:
    print(f"  - {device.device_name}: CFO={device.carrier_freq_offset_ppm:.2f}ppm, "
          f"IQ_Amp={device.iq_imbalance_amplitude:.4f}")

# 选择几种测试调制
test_modulations = MODULATION_LIST[:NUM_TEST_MODULATIONS]
print(f"\n测试调制: {test_modulations}")

# 为每种调制创建TorchSig数据集
print(f"\n创建TorchSig数据集生成器...")
test_signals = {}

for mod_name in test_modulations:
    metadata = DatasetMetadata(
        num_iq_samples_dataset=NUM_IQ_SAMPLES,
        fft_size=512,
        sample_rate=SAMPLE_RATE,
        num_signals_min=1,
        num_signals_max=1,
        snr_db_min=100,  # 高SNR（接近无噪声）
        snr_db_max=100,
        signal_duration_min=0.0004,
        signal_duration_max=0.004,
        signal_bandwidth_min=5e4,
        signal_bandwidth_max=2e5,
        signal_center_freq_min=-4e5,
        signal_center_freq_max=4e5,
        class_list=[mod_name],
        cochannel_overlap_probability=0.0,
    )
    
    dataset = TorchSigIterableDataset(metadata=metadata, seed=42)
    
    # 生成一个样本
    data, label = next(iter(dataset))
    clean_signal = data.numpy() if hasattr(data, 'numpy') else np.array(data)
    clean_signal = normalize_signal_power(clean_signal, target_power=1.0)
    
    test_signals[mod_name] = clean_signal
    print(f"  ✅ {mod_name}: 功率={np.mean(np.abs(clean_signal)**2):.4f}")

# 对每个设备应用RFF
print(f"\n应用RFF损伤...")
rff_signals = {}

for device in devices:
    simulator = RFFImpairmentSimulator(device, SAMPLE_RATE)
    rff_signals[device.device_name] = {}
    
    for mod_name, clean_signal in test_signals.items():
        # 应用RFF
        signal_with_rff = simulator.apply_all_impairments(clean_signal.copy(), center_freq=0.0)
        signal_with_rff = normalize_signal_power(signal_with_rff, target_power=1.0)
        
        # 添加噪声
        final_signal = add_awgn(signal_with_rff, SNR_DB)
        
        rff_signals[device.device_name][mod_name] = {
            'clean': clean_signal,
            'with_rff': signal_with_rff,
            'with_noise': final_signal,
        }

print("✅ RFF损伤应用完成")

# 可视化对比
print("\n生成可视化图表...")

num_devices = len(devices)
num_mods = len(test_modulations)

# 图1：不同设备的同一调制对比（星座图）
fig1, axes = plt.subplots(num_mods, num_devices + 1, figsize=(15, 10))

for mod_idx, mod_name in enumerate(test_modulations):
    # 原始信号
    ax = axes[mod_idx, 0] if num_mods > 1 else axes[0]
    clean = test_signals[mod_name]
    ax.scatter(np.real(clean[::10]), np.imag(clean[::10]), s=1, alpha=0.3)
    ax.set_title(f'{mod_name}\n原始信号', fontsize=9)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    
    # 不同设备的RFF信号
    for dev_idx, device in enumerate(devices):
        ax = axes[mod_idx, dev_idx + 1] if num_mods > 1 else axes[dev_idx + 1]
        signal = rff_signals[device.device_name][mod_name]['with_noise']
        ax.scatter(np.real(signal[::10]), np.imag(signal[::10]), s=1, alpha=0.3)
        ax.set_title(f'{device.device_name}\nSNR={SNR_DB}dB', fontsize=9)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])

plt.suptitle('不同设备的星座图对比', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('test_constellation_comparison.png', dpi=150, bbox_inches='tight')
print("  ✅ 保存: test_constellation_comparison.png")

# 图2：时域和频域对比
fig2, axes = plt.subplots(3, 4, figsize=(16, 10))

mod_name = test_modulations[0]  # 使用第一种调制
device = devices[0]  # 使用第一个设备

signals_to_plot = [
    ('原始信号', test_signals[mod_name]),
    ('RFF (无噪声)', rff_signals[device.device_name][mod_name]['with_rff']),
    ('RFF + 噪声', rff_signals[device.device_name][mod_name]['with_noise']),
]

for row, (title, signal) in enumerate(signals_to_plot):
    # 时域 - I分量
    ax = axes[row, 0]
    ax.plot(np.real(signal[:500]), linewidth=0.5)
    ax.set_title(f'{title}\nI分量 (时域)')
    ax.set_xlabel('样本')
    ax.set_ylabel('幅度')
    ax.grid(True, alpha=0.3)
    
    # 时域 - Q分量
    ax = axes[row, 1]
    ax.plot(np.imag(signal[:500]), linewidth=0.5)
    ax.set_title(f'Q分量 (时域)')
    ax.set_xlabel('样本')
    ax.set_ylabel('幅度')
    ax.grid(True, alpha=0.3)
    
    # 幅度
    ax = axes[row, 2]
    ax.plot(np.abs(signal[:500]), linewidth=0.5)
    ax.set_title(f'幅度')
    ax.set_xlabel('样本')
    ax.set_ylabel('|IQ|')
    ax.grid(True, alpha=0.3)
    
    # 频谱
    ax = axes[row, 3]
    fft = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/SAMPLE_RATE))
    ax.plot(freqs/1e3, 20*np.log10(np.abs(fft) + 1e-10), linewidth=0.5)
    ax.set_title(f'功率谱')
    ax.set_xlabel('频率 (kHz)')
    ax.set_ylabel('功率 (dB)')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'时域和频域对比 - {mod_name} - {device.device_name}', fontsize=14)
plt.tight_layout()
plt.savefig('test_time_freq_comparison.png', dpi=150, bbox_inches='tight')
print("  ✅ 保存: test_time_freq_comparison.png")

# 图3：RFF特征差异可视化
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

mod_name = test_modulations[0]

# 不同设备的星座图叠加
ax = axes[0, 0]
for device in devices:
    signal = rff_signals[device.device_name][mod_name]['with_rff']
    ax.scatter(np.real(signal[::20]), np.imag(signal[::20]), 
              s=5, alpha=0.4, label=device.device_name)
ax.set_title('不同设备星座图叠加 (无噪声)')
ax.set_xlabel('I')
ax.set_ylabel('Q')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# 不同设备的频谱叠加
ax = axes[0, 1]
for device in devices:
    signal = rff_signals[device.device_name][mod_name]['with_rff']
    fft = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/SAMPLE_RATE))
    ax.plot(freqs/1e3, 20*np.log10(np.abs(fft) + 1e-10), 
           alpha=0.7, linewidth=1, label=device.device_name)
ax.set_title('不同设备功率谱叠加')
ax.set_xlabel('频率 (kHz)')
ax.set_ylabel('功率 (dB)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 幅度分布
ax = axes[1, 0]
for device in devices:
    signal = rff_signals[device.device_name][mod_name]['with_rff']
    ax.hist(np.abs(signal), bins=50, alpha=0.5, label=device.device_name)
ax.set_title('幅度分布')
ax.set_xlabel('幅度')
ax.set_ylabel('频数')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 相位分布
ax = axes[1, 1]
for device in devices:
    signal = rff_signals[device.device_name][mod_name]['with_rff']
    ax.hist(np.angle(signal), bins=50, alpha=0.5, label=device.device_name)
ax.set_title('相位分布')
ax.set_xlabel('相位 (弧度)')
ax.set_ylabel('频数')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f'RFF特征差异可视化 - {mod_name}', fontsize=14)
plt.tight_layout()
plt.savefig('test_rff_features.png', dpi=150, bbox_inches='tight')
print("  ✅ 保存: test_rff_features.png")

# 打印统计信息
print("\n" + "="*80)
print("统计信息:")
print("="*80)

for device in devices:
    print(f"\n{device.device_name}:")
    print(f"  RFF参数:")
    print(f"    - I/Q幅度不平衡: {device.iq_imbalance_amplitude:.4f}")
    print(f"    - I/Q相位不平衡: {device.iq_imbalance_phase_deg:.2f}°")
    print(f"    - 载波频偏: {device.carrier_freq_offset_ppm:.2f} ppm")
    print(f"    - 相位噪声: {device.phase_noise_level:.4f}")
    print(f"    - DC偏移: ({device.dc_offset_i:.4f}, {device.dc_offset_q:.4f})")
    
    for mod_name in test_modulations:
        signal = rff_signals[device.device_name][mod_name]['with_rff']
        power = np.mean(np.abs(signal)**2)
        print(f"  {mod_name}: 功率={power:.4f}")

print("\n" + "="*80)
print("✅ 测试完成！")
print("="*80)
print("\n生成的图片:")
print("  1. test_constellation_comparison.png - 不同设备星座图对比")
print("  2. test_time_freq_comparison.png - 时域频域对比")
print("  3. test_rff_features.png - RFF特征差异")
print("\n下一步：运行 generate_dataset.py 生成完整数据集")
print("="*80)

