#!/usr/bin/env python
"""
原始配置备份 - 雷达RFF数据集生成参数
备份时间：2025-10-20
备份原因：更新为TorchSig宽带默认值前的原始配置
"""

# ============================================================================
# 原始配置参数 (更新前)
# ============================================================================

class OriginalDatasetConfig:
    """原始数据集生成配置 - 备份"""
    
    # 设备和调制参数
    NUM_DEVICES = 20
    NUM_MODULATIONS = 26
    
    # 样本数量
    SAMPLES_PER_DEVICE_MODULATION = 1000  # 每个（设备-调制）组合的样本数
    
    # 信号参数 - 原始配置
    NUM_IQ_SAMPLES = 4096     # 原始：4096个IQ采样点
    SAMPLE_RATE = 1e6         # 原始：1 MHz采样率
    FFT_SIZE = 512            # 原始：512点FFT
    
    # SNR配置 - 原始
    FIXED_SNR_DB = None       # 设置为None则使用随机SNR范围
    SNR_MIN_DB = 0.0          # 原始：0 dB
    SNR_MAX_DB = 30.0         # 原始：30 dB
    
    # 信号生成参数 - 原始配置
    SIGNAL_DURATION_MIN = 0.0004   # 原始：0.4 ms
    SIGNAL_DURATION_MAX = 0.004    # 原始：4 ms
    SIGNAL_BANDWIDTH_MIN = 5e4     # 原始：50 kHz
    SIGNAL_BANDWIDTH_MAX = 2e5     # 原始：200 kHz
    SIGNAL_CENTER_FREQ_MIN = -4e5  # 原始：-400 kHz
    SIGNAL_CENTER_FREQ_MAX = 4e5   # 原始：+400 kHz
    
    # RFF参数
    USE_DIVERSE_RFF = True    # True=差异明显（易分类），False=真实分布（更难）
    RFF_SEED = 42
    
    # 输出设置
    OUTPUT_DIR = "./radar_rff_dataset"
    USE_HDF5 = True           # True=HDF5格式（推荐），False=numpy格式
    
    # 并行处理
    NUM_WORKERS = 8           # 原始配置
    
    # 随机种子
    RANDOM_SEED = 12345
    
    @classmethod
    def total_samples(cls):
        """计算总样本数"""
        return cls.NUM_DEVICES * cls.NUM_MODULATIONS * cls.SAMPLES_PER_DEVICE_MODULATION
    
    @classmethod
    def total_classes(cls):
        """计算总类别数"""
        return cls.NUM_DEVICES * cls.NUM_MODULATIONS
    
    @classmethod
    def estimated_size_gb(cls):
        """估算数据集大小（GB）- 原始配置"""
        # complex64: 8 bytes per sample
        bytes_per_sample = cls.NUM_IQ_SAMPLES * 8
        total_bytes = cls.total_samples() * bytes_per_sample
        return total_bytes / (1024**3)

# ============================================================================
# 新配置参数 (TorchSig宽带默认值)
# ============================================================================

class TorchSigWidebandConfig:
    """TorchSig宽带默认配置"""
    
    # 设备和调制参数
    NUM_DEVICES = 20
    NUM_MODULATIONS = 26
    
    # 样本数量
    SAMPLES_PER_DEVICE_MODULATION = 1000
    
    # 信号参数 - TorchSig宽带默认值
    NUM_IQ_SAMPLES = 1048576  # 1024^2 (TorchSig宽带默认)
    SAMPLE_RATE = 100e6       # 100 MHz (TorchSig宽带默认)
    FFT_SIZE = 1024           # TorchSig宽带默认
    
    # SNR配置 - TorchSig默认
    FIXED_SNR_DB = None
    SNR_MIN_DB = 0.0          # TorchSig默认
    SNR_MAX_DB = 50.0         # TorchSig默认
    
    # 信号生成参数 - TorchSig宽带默认值
    SIGNAL_DURATION_MIN = 0.000131072  # 0.05*num_iq_samples_dataset/sample_rate = 0.13 ms
    SIGNAL_DURATION_MAX = 0.000262144  # 0.10*num_iq_samples_dataset/sample_rate = 0.26 ms
    SIGNAL_BANDWIDTH_MIN = 5e6         # sample_rate/20 = 5 MHz
    SIGNAL_BANDWIDTH_MAX = 10e6        # sample_rate/10 = 10 MHz
    SIGNAL_CENTER_FREQ_MIN = -50e6     # -sample_rate/2 = -50 MHz
    SIGNAL_CENTER_FREQ_MAX = 49999999  # sample_rate/2 - 1 = 49.999999 MHz
    
    @classmethod
    def estimated_size_gb(cls):
        """估算数据集大小（GB）- TorchSig宽带配置"""
        # complex64: 8 bytes per sample
        bytes_per_sample = cls.NUM_IQ_SAMPLES * 8
        total_bytes = (cls.NUM_DEVICES * cls.NUM_MODULATIONS * 
                      cls.SAMPLES_PER_DEVICE_MODULATION * bytes_per_sample)
        return total_bytes / (1024**3)

# ============================================================================
# 配置对比
# ============================================================================

def print_config_comparison():
    """打印原始配置与新配置的对比"""
    print("=" * 80)
    print("配置对比：原始配置 vs TorchSig宽带默认配置")
    print("=" * 80)
    
    print(f"{'参数':<25} {'原始配置':<20} {'TorchSig宽带配置':<20} {'变化倍数':<15}")
    print("-" * 80)
    
    # IQ采样点数
    original_samples = OriginalDatasetConfig.NUM_IQ_SAMPLES
    new_samples = TorchSigWidebandConfig.NUM_IQ_SAMPLES
    print(f"{'IQ采样点数':<25} {original_samples:<20} {new_samples:<20} {new_samples/original_samples:.1f}x")
    
    # 采样率
    original_sr = OriginalDatasetConfig.SAMPLE_RATE / 1e6
    new_sr = TorchSigWidebandConfig.SAMPLE_RATE / 1e6
    print(f"{'采样率 (MHz)':<25} {original_sr:<20} {new_sr:<20} {new_sr/original_sr:.1f}x")
    
    # 信号带宽
    original_bw = f"{OriginalDatasetConfig.SIGNAL_BANDWIDTH_MIN/1e3:.0f}-{OriginalDatasetConfig.SIGNAL_BANDWIDTH_MAX/1e3:.0f} kHz"
    new_bw = f"{TorchSigWidebandConfig.SIGNAL_BANDWIDTH_MIN/1e6:.0f}-{TorchSigWidebandConfig.SIGNAL_BANDWIDTH_MAX/1e6:.0f} MHz"
    print(f"{'信号带宽':<25} {original_bw:<20} {new_bw:<20} {'100x':<15}")
    
    # 数据集大小
    original_size = OriginalDatasetConfig.estimated_size_gb()
    new_size = TorchSigWidebandConfig.estimated_size_gb()
    print(f"{'数据集大小 (GB)':<25} {original_size:.1f} GB{'':<15} {new_size:.1f} GB{'':<15} {new_size/original_size:.1f}x")
    
    print("=" * 80)
    print("⚠️  注意：新配置将产生极大的数据集！")
    print(f"   单个SNR数据集：~{new_size:.1f} GB")
    print(f"   11个SNR数据集总计：~{new_size*11:.1f} GB")
    print("=" * 80)

if __name__ == "__main__":
    print_config_comparison()
