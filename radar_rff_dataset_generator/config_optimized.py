#!/usr/bin/env python
"""
优化配置 - 平衡存储空间与信号质量
在宽带处理能力和存储需求之间找到平衡点
"""

class OptimizedDatasetConfig:
    """优化的数据集配置 - 平衡版本"""
    
    # 设备和调制参数
    NUM_DEVICES = 20
    NUM_MODULATIONS = 26
    
    # 样本数量
    SAMPLES_PER_DEVICE_MODULATION = 1000
    
    # 信号参数 - 优化配置（存储空间减少16倍）
    NUM_IQ_SAMPLES = 65536    # 64K点（比1M点小16倍，比4K点大16倍）
    SAMPLE_RATE = 100e6       # 保持100 MHz采样率
    FFT_SIZE = 1024           # 保持1024点FFT
    
    # SNR配置
    FIXED_SNR_DB = None
    SNR_MIN_DB = 0.0
    SNR_MAX_DB = 50.0
    
    # 信号生成参数 - 调整到适合64K采样点
    SIGNAL_DURATION_MIN = 0.000032768   # 0.05*65536/100e6 = 0.033 ms
    SIGNAL_DURATION_MAX = 0.000065536   # 0.10*65536/100e6 = 0.066 ms
    SIGNAL_BANDWIDTH_MIN = 5e6          # 5 MHz
    SIGNAL_BANDWIDTH_MAX = 10e6         # 10 MHz  
    SIGNAL_CENTER_FREQ_MIN = -50e6      # -50 MHz
    SIGNAL_CENTER_FREQ_MAX = 49999999   # +50 MHz
    
    # RFF参数
    USE_DIVERSE_RFF = True
    RFF_SEED = 42
    
    # 输出设置
    OUTPUT_DIR = "./radar_rff_dataset_optimized"
    USE_HDF5 = True
    
    # 并行处理
    NUM_WORKERS = 8
    
    # 随机种子
    RANDOM_SEED = 12345
    
    @classmethod
    def total_samples(cls):
        return cls.NUM_DEVICES * cls.NUM_MODULATIONS * cls.SAMPLES_PER_DEVICE_MODULATION
    
    @classmethod
    def estimated_size_gb(cls):
        """估算数据集大小（GB）"""
        bytes_per_sample = cls.NUM_IQ_SAMPLES * 8  # complex64
        total_bytes = cls.total_samples() * bytes_per_sample
        return total_bytes / (1024**3)
    
    @classmethod
    def get_analysis(cls):
        """分析此配置的特点"""
        time_window = cls.NUM_IQ_SAMPLES / cls.SAMPLE_RATE * 1000  # ms
        freq_resolution = cls.SAMPLE_RATE / cls.NUM_IQ_SAMPLES / 1000  # kHz
        
        return {
            'time_window_ms': time_window,
            'freq_resolution_khz': freq_resolution,
            'size_gb': cls.estimated_size_gb(),
            'samples_per_signal': cls.NUM_IQ_SAMPLES,
        }


class ConservativeDatasetConfig:
    """保守配置 - 最小化存储需求"""
    
    # 设备和调制参数
    NUM_DEVICES = 20
    NUM_MODULATIONS = 26
    
    # 样本数量
    SAMPLES_PER_DEVICE_MODULATION = 1000
    
    # 信号参数 - 保守配置（存储空间减少64倍）
    NUM_IQ_SAMPLES = 16384     # 16K点
    SAMPLE_RATE = 50e6         # 降低到50 MHz采样率
    FFT_SIZE = 512             # 512点FFT
    
    # SNR配置
    FIXED_SNR_DB = None
    SNR_MIN_DB = 0.0
    SNR_MAX_DB = 50.0
    
    # 信号生成参数 - 调整到适合16K采样点和50MHz采样率
    SIGNAL_DURATION_MIN = 0.000016384   # 0.05*16384/50e6 = 0.016 ms
    SIGNAL_DURATION_MAX = 0.000032768   # 0.10*16384/50e6 = 0.033 ms
    SIGNAL_BANDWIDTH_MIN = 2.5e6        # sample_rate/20 = 2.5 MHz
    SIGNAL_BANDWIDTH_MAX = 5e6          # sample_rate/10 = 5 MHz
    SIGNAL_CENTER_FREQ_MIN = -25e6      # -sample_rate/2 = -25 MHz
    SIGNAL_CENTER_FREQ_MAX = 24999999   # sample_rate/2 - 1 = 25 MHz
    
    # RFF参数
    USE_DIVERSE_RFF = True
    RFF_SEED = 42
    
    # 输出设置
    OUTPUT_DIR = "./radar_rff_dataset_conservative"
    USE_HDF5 = True
    
    # 并行处理
    NUM_WORKERS = 8
    
    # 随机种子
    RANDOM_SEED = 12345
    
    @classmethod
    def total_samples(cls):
        return cls.NUM_DEVICES * cls.NUM_MODULATIONS * cls.SAMPLES_PER_DEVICE_MODULATION
    
    @classmethod
    def estimated_size_gb(cls):
        """估算数据集大小（GB）"""
        bytes_per_sample = cls.NUM_IQ_SAMPLES * 8  # complex64
        total_bytes = cls.total_samples() * bytes_per_sample
        return total_bytes / (1024**3)
    
    @classmethod
    def get_analysis(cls):
        """分析此配置的特点"""
        time_window = cls.NUM_IQ_SAMPLES / cls.SAMPLE_RATE * 1000  # ms
        freq_resolution = cls.SAMPLE_RATE / cls.NUM_IQ_SAMPLES / 1000  # kHz
        
        return {
            'time_window_ms': time_window,
            'freq_resolution_khz': freq_resolution,
            'size_gb': cls.estimated_size_gb(),
            'samples_per_signal': cls.NUM_IQ_SAMPLES,
        }


def compare_all_configs():
    """对比所有配置方案"""
    from config_backup_original import OriginalDatasetConfig, TorchSigWidebandConfig
    
    configs = {
        '原始配置': OriginalDatasetConfig,
        'TorchSig宽带': TorchSigWidebandConfig,
        '优化配置': OptimizedDatasetConfig,
        '保守配置': ConservativeDatasetConfig,
    }
    
    print("=" * 100)
    print("所有配置方案对比")
    print("=" * 100)
    print(f"{'配置名称':<15} {'采样点数':<10} {'采样率':<10} {'数据集大小':<12} {'时间窗口':<10} {'频率分辨率':<12}")
    print("-" * 100)
    
    for name, config in configs.items():
        if hasattr(config, 'get_analysis'):
            analysis = config.get_analysis()
            print(f"{name:<15} {analysis['samples_per_signal']:<10} "
                  f"{config.SAMPLE_RATE/1e6:.0f} MHz{'':<4} "
                  f"{analysis['size_gb']:.1f} GB{'':<7} "
                  f"{analysis['time_window_ms']:.2f} ms{'':<5} "
                  f"{analysis['freq_resolution_khz']:.1f} kHz")
        else:
            # 原始配置的计算
            if name == '原始配置':
                time_window = config.NUM_IQ_SAMPLES / config.SAMPLE_RATE * 1000
                freq_res = config.SAMPLE_RATE / config.NUM_IQ_SAMPLES / 1000
                size_gb = config.estimated_size_gb()
                print(f"{name:<15} {config.NUM_IQ_SAMPLES:<10} "
                      f"{config.SAMPLE_RATE/1e6:.0f} MHz{'':<4} "
                      f"{size_gb:.1f} GB{'':<7} "
                      f"{time_window:.2f} ms{'':<5} "
                      f"{freq_res:.1f} kHz")
            elif name == 'TorchSig宽带':
                time_window = config.NUM_IQ_SAMPLES / config.SAMPLE_RATE * 1000
                freq_res = config.SAMPLE_RATE / config.NUM_IQ_SAMPLES / 1000
                size_gb = config.estimated_size_gb()
                print(f"{name:<15} {config.NUM_IQ_SAMPLES:<10} "
                      f"{config.SAMPLE_RATE/1e6:.0f} MHz{'':<4} "
                      f"{size_gb:.1f} GB{'':<7} "
                      f"{time_window:.2f} ms{'':<5} "
                      f"{freq_res:.1f} kHz")
    
    print("=" * 100)
    print("推荐选择：")
    print("  🎯 优化配置：平衡性能与存储，适合大多数应用")
    print("  💾 保守配置：最小存储需求，适合资源受限环境")
    print("  🚀 TorchSig宽带：最佳性能，适合充足存储环境")
    print("=" * 100)


if __name__ == "__main__":
    compare_all_configs()
