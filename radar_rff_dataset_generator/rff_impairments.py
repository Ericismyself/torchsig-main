"""
射频指纹（RFF）硬件损伤模拟器

基于TorchSig的变换功能，实现各种硬件非理想性的模拟。

作者：根据研究需求生成
日期：2025-10-18
"""

import numpy as np
import torch
from typing import Optional, Tuple
from config_rff_parameters import RFFParameters


class RFFImpairmentSimulator:
    """
    射频指纹硬件损伤模拟器
    
    为信号添加设备特定的硬件非理想性，模拟真实雷达设备的RFF特征。
    """
    
    def __init__(self, rff_params: RFFParameters, sample_rate: float = 1e6):
        """
        初始化RFF损伤模拟器
        
        参数:
            rff_params: 设备的RFF参数配置
            sample_rate: 采样率 (Hz)
        """
        self.rff_params = rff_params
        self.sample_rate = sample_rate
    
    def apply_iq_imbalance(self, signal: np.ndarray) -> np.ndarray:
        """
        应用I/Q不平衡
        
        I/Q不平衡是指I和Q两路信号的幅度和相位不完全正交，
        主要由模拟混频器和滤波器的不匹配引起。
        
        参数:
            signal: 输入复数信号 (IQ数据)
        
        返回:
            带有I/Q不平衡的信号
        """
        # 提取I和Q分量
        I = np.real(signal)
        Q = np.imag(signal)
        
        # 应用幅度不平衡
        amp_imbalance = self.rff_params.iq_imbalance_amplitude
        I_imb = I * amp_imbalance
        
        # 应用相位不平衡
        phase_imbalance_rad = np.deg2rad(self.rff_params.iq_imbalance_phase_deg)
        Q_imb = Q * np.cos(phase_imbalance_rad) + I * np.sin(phase_imbalance_rad)
        
        # 重组复数信号
        signal_imbalanced = I_imb + 1j * Q_imb
        
        return signal_imbalanced
    
    def apply_carrier_freq_offset(self, signal: np.ndarray, 
                                   center_freq: float = 0.0) -> np.ndarray:
        """
        应用载波频率偏移（CFO）
        
        CFO由本振晶振的频率误差引起，不同设备的晶振精度不同。
        
        参数:
            signal: 输入复数信号
            center_freq: 信号中心频率 (Hz)，默认为0（基带）
        
        返回:
            带有频率偏移的信号
        """
        # 将ppm转换为实际频率偏移
        freq_offset_hz = (center_freq * self.rff_params.carrier_freq_offset_ppm / 1e6)
        
        # 生成时间向量
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        # 应用频率偏移（复数旋转）
        phase_shift = 2 * np.pi * freq_offset_hz * t
        freq_shifted_signal = signal * np.exp(1j * phase_shift)
        
        return freq_shifted_signal
    
    def apply_sampling_rate_offset(self, signal: np.ndarray) -> np.ndarray:
        """
        应用采样率偏移（SRO）
        
        SRO由ADC采样时钟的频率误差引起。
        注意：实际实现采样率偏移需要重采样，这里用简化的相位累积模拟。
        
        参数:
            signal: 输入复数信号
        
        返回:
            带有采样率偏移影响的信号
        """
        # 采样率偏移会导致相位的二次累积
        # 这里使用简化模型：线性相位误差
        sro_ppm = self.rff_params.sampling_rate_offset_ppm
        
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        # 采样率偏移导致的相位误差（简化模型）
        # 实际SRO效果更复杂，涉及符号时钟偏移
        phase_error = np.pi * sro_ppm / 1e6 * t * t  # 二次相位误差
        
        sro_signal = signal * np.exp(1j * phase_error)
        
        return sro_signal
    
    def apply_phase_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        应用相位噪声
        
        相位噪声由振荡器的短期频率不稳定性引起，
        表现为载波相位的随机抖动。
        
        参数:
            signal: 输入复数信号
        
        返回:
            带有相位噪声的信号
        """
        num_samples = len(signal)
        
        # 生成高斯白相位噪声
        # 实际相位噪声应该是有色噪声（低频成分更强），这里简化为白噪声
        phase_noise_std = self.rff_params.phase_noise_level
        phase_noise = np.random.normal(0, phase_noise_std, num_samples)
        
        # 可选：应用低通滤波使相位噪声更真实（有色噪声）
        # 这里暂时使用白噪声模型
        
        # 应用相位噪声
        noisy_signal = signal * np.exp(1j * phase_noise)
        
        return noisy_signal
    
    def apply_pa_nonlinearity(self, signal: np.ndarray) -> np.ndarray:
        """
        应用功率放大器非线性
        
        功放在大信号时会出现饱和，导致幅度压缩和相位失真。
        使用Rapp模型模拟AM/AM特性。
        
        参数:
            signal: 输入复数信号
        
        返回:
            经过功放非线性的信号
        """
        # Rapp模型参数
        p = self.rff_params.pa_rapp_smoothness  # 平滑因子（越大越陡）
        sat_level = self.rff_params.pa_saturation_level  # 饱和电平
        
        # 计算输入幅度
        input_amp = np.abs(signal)
        
        # Rapp模型: 输出幅度 = 输入幅度 / (1 + (输入幅度/饱和电平)^(2p))^(1/(2p))
        output_amp = input_amp / np.power(
            1 + np.power(input_amp / sat_level, 2 * p),
            1 / (2 * p)
        )
        
        # 保持原始相位（AM/AM特性，忽略AM/PM）
        # 实际功放还有AM/PM效应，可以后续添加
        phase = np.angle(signal)
        
        # 重构信号
        pa_signal = output_amp * np.exp(1j * phase)
        
        return pa_signal
    
    def apply_dc_offset(self, signal: np.ndarray) -> np.ndarray:
        """
        应用DC偏移
        
        DC偏移由模拟电路的偏置引起，表现为I和Q通道的直流分量。
        
        参数:
            signal: 输入复数信号
        
        返回:
            带有DC偏移的信号
        """
        dc_offset = (self.rff_params.dc_offset_i + 
                     1j * self.rff_params.dc_offset_q)
        
        return signal + dc_offset
    
    def apply_all_impairments(self, 
                              signal: np.ndarray,
                              center_freq: float = 0.0,
                              apply_order: Optional[list] = None) -> np.ndarray:
        """
        按顺序应用所有RFF损伤
        
        参数:
            signal: 输入复数信号
            center_freq: 信号中心频率
            apply_order: 损伤应用顺序列表，默认为标准顺序
        
        返回:
            应用所有RFF损伤后的信号
        """
        if apply_order is None:
            # 默认顺序（模拟信号在发射机中的传播路径）
            apply_order = [
                'iq_imbalance',
                'dc_offset',
                'carrier_freq_offset',
                'sampling_rate_offset',
                'phase_noise',
                'pa_nonlinearity',
            ]
        
        result = signal.copy()
        
        for impairment in apply_order:
            if impairment == 'iq_imbalance':
                result = self.apply_iq_imbalance(result)
            elif impairment == 'dc_offset':
                result = self.apply_dc_offset(result)
            elif impairment == 'carrier_freq_offset':
                result = self.apply_carrier_freq_offset(result, center_freq)
            elif impairment == 'sampling_rate_offset':
                result = self.apply_sampling_rate_offset(result)
            elif impairment == 'phase_noise':
                result = self.apply_phase_noise(result)
            elif impairment == 'pa_nonlinearity':
                result = self.apply_pa_nonlinearity(result)
            else:
                print(f"警告: 未知的损伤类型 '{impairment}'，已跳过")
        
        return result
    
    def get_device_info(self) -> dict:
        """返回设备信息"""
        return {
            'device_id': self.rff_params.device_id,
            'device_name': self.rff_params.device_name,
            'rff_parameters': self.rff_params.to_dict(),
        }


# ============================================================================
# 工具函数
# ============================================================================

def normalize_signal_power(signal: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """
    归一化信号功率
    
    参数:
        signal: 输入信号
        target_power: 目标平均功率
    
    返回:
        归一化后的信号
    """
    current_power = np.mean(np.abs(signal) ** 2)
    scale_factor = np.sqrt(target_power / current_power)
    return signal * scale_factor


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    添加加性高斯白噪声（AWGN）
    
    参数:
        signal: 输入信号
        snr_db: 目标信噪比 (dB)
    
    返回:
        加噪后的信号
    """
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # 生成复高斯白噪声
    noise_std = np.sqrt(noise_power / 2)  # /2 因为I和Q各占一半功率
    noise = (np.random.normal(0, noise_std, len(signal)) + 
             1j * np.random.normal(0, noise_std, len(signal)))
    
    return signal + noise


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config_rff_parameters import generate_device_parameters
    
    print("="*80)
    print("RFF硬件损伤模拟器测试")
    print("="*80)
    
    # 生成测试设备
    devices = generate_device_parameters(num_devices=3, seed=42, diverse=True)
    
    # 生成测试信号（QPSK为例）
    num_samples = 4096
    sample_rate = 1e6
    
    # 简单的QPSK信号（4个符号重复）
    symbols = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
    num_symbols = num_samples // (num_samples // 4)
    test_signal_clean = np.repeat(symbols, num_samples // 4)[:num_samples]
    
    # 归一化
    test_signal_clean = normalize_signal_power(test_signal_clean, target_power=1.0)
    
    print(f"\n测试信号: {num_samples} 个IQ样本, 采样率 {sample_rate/1e6} MHz")
    print(f"信号功率: {np.mean(np.abs(test_signal_clean)**2):.4f}")
    
    # 对3个不同设备应用RFF损伤
    signals_with_rff = {}
    
    for i, device in enumerate(devices):
        print(f"\n处理 {device.device_name}...")
        
        simulator = RFFImpairmentSimulator(device, sample_rate)
        
        # 应用所有损伤
        signal_with_rff = simulator.apply_all_impairments(
            test_signal_clean.copy(),
            center_freq=0.0  # 基带信号
        )
        
        # 添加噪声
        snr_db = 20
        signal_with_noise = add_awgn(signal_with_rff, snr_db)
        
        signals_with_rff[device.device_name] = {
            'clean': test_signal_clean,
            'with_rff': signal_with_rff,
            'with_noise': signal_with_noise,
        }
        
        print(f"  - 应用RFF后功率: {np.mean(np.abs(signal_with_rff)**2):.4f}")
        print(f"  - 添加噪声后(SNR={snr_db}dB)功率: {np.mean(np.abs(signal_with_noise)**2):.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, (device_name, signals) in enumerate(signals_with_rff.items()):
        # 星座图对比
        ax = axes[i, 0]
        ax.scatter(np.real(signals['clean'][::10]), 
                  np.imag(signals['clean'][::10]),
                  alpha=0.5, s=10, label='原始')
        ax.scatter(np.real(signals['with_rff'][::10]), 
                  np.imag(signals['with_rff'][::10]),
                  alpha=0.5, s=10, label='RFF')
        ax.set_title(f'{device_name}\n星座图')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 时域波形
        ax = axes[i, 1]
        ax.plot(np.abs(signals['clean'][:200]), alpha=0.7, label='原始')
        ax.plot(np.abs(signals['with_rff'][:200]), alpha=0.7, label='RFF')
        ax.set_title('幅度时域波形 (前200点)')
        ax.set_xlabel('样本')
        ax.set_ylabel('幅度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 频谱
        ax = axes[i, 2]
        fft_clean = np.fft.fftshift(np.fft.fft(signals['clean']))
        fft_rff = np.fft.fftshift(np.fft.fft(signals['with_rff']))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signals['clean']), 1/sample_rate))
        
        ax.plot(freqs/1e3, 20*np.log10(np.abs(fft_clean)), alpha=0.7, label='原始')
        ax.plot(freqs/1e3, 20*np.log10(np.abs(fft_rff)), alpha=0.7, label='RFF')
        ax.set_title('功率谱')
        ax.set_xlabel('频率 (kHz)')
        ax.set_ylabel('功率 (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rff_impairments_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存至: rff_impairments_comparison.png")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)

