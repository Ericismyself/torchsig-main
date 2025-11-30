#!/usr/bin/env python
"""
雷达RFF数据集生成主脚本

生成包含20类雷达设备 × 26种调制方式 = 520类信号的数据集

作者：根据研究需求生成
日期：2025-10-18
"""

import os
import sys
import numpy as np
import h5py
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime

# 导入TorchSig
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.datasets.dataset_metadata import DatasetMetadata

# 导入本地模块
from config_rff_parameters import (
    generate_device_parameters, 
    MODULATION_LIST, 
    get_combined_class_id,
    RFFParameters
)
from rff_impairments import RFFImpairmentSimulator, add_awgn, normalize_signal_power


# ============================================================================
# 配置参数
# ============================================================================

class DatasetConfig:
    """数据集生成配置"""
    
    # 设备和调制参数
    NUM_DEVICES = 20
    NUM_MODULATIONS = 26
    MODULATION_LIST = MODULATION_LIST
    
    # 样本数量
    SAMPLES_PER_DEVICE_MODULATION = 1000  # 每个（设备-调制）组合的样本数
    
    # 信号参数 - 使用TorchSig宽带默认值
    NUM_IQ_SAMPLES = 1048576  # 1024^2 (TorchSig宽带默认)
    SAMPLE_RATE = 100e6  # 100 MHz (TorchSig宽带默认)
    FFT_SIZE = 1024      # TorchSig宽带默认
    
    # SNR配置
    # 模式1: 随机SNR范围 - 设置 FIXED_SNR_DB = None
    # 模式2: 固定SNR - 设置 FIXED_SNR_DB = 具体值（如5, 10, 15, 20, 25）
    FIXED_SNR_DB = None  # 设置为None则使用随机SNR范围，设置为数值则使用固定SNR
    SNR_MIN_DB = 0.0     # TorchSig默认
    SNR_MAX_DB = 50.0    # TorchSig默认 (从30提升到50)
    
    # 信号生成参数 - 使用TorchSig宽带默认值
    SIGNAL_DURATION_MIN = 0.000131072  # 0.05*num_iq_samples_dataset/sample_rate = 0.13 ms
    SIGNAL_DURATION_MAX = 0.000262144  # 0.10*num_iq_samples_dataset/sample_rate = 0.26 ms
    SIGNAL_BANDWIDTH_MIN = 5e6    # sample_rate/20 = 5 MHz
    SIGNAL_BANDWIDTH_MAX = 10e6   # sample_rate/10 = 10 MHz
    SIGNAL_CENTER_FREQ_MIN = -50e6  # -sample_rate/2 = -50 MHz
    SIGNAL_CENTER_FREQ_MAX = 49999999   # sample_rate/2 - 1 = 49.999999 MHz
    
    # RFF参数
    USE_DIVERSE_RFF = True  # True=差异明显（易分类），False=真实分布（更难）
    RFF_SEED = 42
    
    # 输出设置
    OUTPUT_DIR = "./radar_rff_dataset"
    USE_HDF5 = True  # True=HDF5格式（推荐），False=numpy格式
    
    # 并行处理
    NUM_WORKERS = max(1, cpu_count() - 2)  # 保留2个核心给系统
    
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
        """估算数据集大小（GB）"""
        # complex64: 8 bytes per sample
        bytes_per_sample = cls.NUM_IQ_SAMPLES * 8
        total_bytes = cls.total_samples() * bytes_per_sample
        return total_bytes / (1024**3)


# ============================================================================
# 数据集生成器
# ============================================================================

class RadarRFFDatasetGenerator:
    """雷达RFF数据集生成器"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # 生成设备RFF参数
        print("生成设备RFF参数...")
        self.devices = generate_device_parameters(
            num_devices=config.NUM_DEVICES,
            seed=config.RFF_SEED,
            diverse=config.USE_DIVERSE_RFF
        )
        
        # 为每个设备创建RFF模拟器
        self.rff_simulators = {
            device.device_id: RFFImpairmentSimulator(device, config.SAMPLE_RATE)
            for device in self.devices
        }
        
        # 创建TorchSig数据集生成器（每种调制一个）
        self.torchsig_datasets = {}
        self._create_torchsig_datasets()
        
        print(f"✅ 初始化完成:")
        print(f"   - 设备数: {config.NUM_DEVICES}")
        print(f"   - 调制数: {config.NUM_MODULATIONS}")
        print(f"   - 总类别数: {config.total_classes()}")
        print(f"   - 总样本数: {config.total_samples()}")
        print(f"   - 预计大小: {config.estimated_size_gb():.2f} GB")
    
    def _create_torchsig_datasets(self):
        """为每种调制创建TorchSig数据集"""
        print(f"\n创建TorchSig数据集生成器（{self.config.NUM_MODULATIONS}种调制）...")
        
        for mod_id, mod_name in enumerate(tqdm(self.config.MODULATION_LIST, 
                                                desc="创建数据集")):
            # 为每种调制创建独立的元数据
            metadata = DatasetMetadata(
                num_iq_samples_dataset=self.config.NUM_IQ_SAMPLES,
                fft_size=self.config.FFT_SIZE,
                sample_rate=self.config.SAMPLE_RATE,
                
                # 单信号模式
                num_signals_min=1,
                num_signals_max=1,
                
                # SNR设置（先生成干净信号，后续手动添加噪声）
                snr_db_min=100,  # 高SNR，近似无噪声
                snr_db_max=100,
                
                # 信号参数
                signal_duration_min=self.config.SIGNAL_DURATION_MIN,
                signal_duration_max=self.config.SIGNAL_DURATION_MAX,
                signal_bandwidth_min=self.config.SIGNAL_BANDWIDTH_MIN,
                signal_bandwidth_max=self.config.SIGNAL_BANDWIDTH_MAX,
                signal_center_freq_min=self.config.SIGNAL_CENTER_FREQ_MIN,
                signal_center_freq_max=self.config.SIGNAL_CENTER_FREQ_MAX,
                
                # 只生成这一种调制
                class_list=[mod_name],
                cochannel_overlap_probability=0.0,
            )
            
            # 创建数据集
            dataset = TorchSigIterableDataset(
                metadata=metadata,
                seed=self.config.RANDOM_SEED + mod_id,  # 每种调制不同种子
            )
            
            self.torchsig_datasets[mod_name] = {
                'dataset': dataset,
                'metadata': metadata,
                'mod_id': mod_id,
            }
    
    def generate_single_sample(self, 
                               device_id: int, 
                               mod_name: str,
                               snr_db: float) -> Tuple[np.ndarray, Dict]:
        """
        生成单个样本
        
        参数:
            device_id: 设备ID (0-19)
            mod_name: 调制名称
            snr_db: 信噪比 (dB)
        
        返回:
            (signal, metadata) 元组
        """
        # 从TorchSig获取基础调制信号
        dataset_info = self.torchsig_datasets[mod_name]
        dataset = dataset_info['dataset']
        mod_id = dataset_info['mod_id']
        
        # 获取一个信号样本
        # 注意：TorchSig的迭代器每次调用都会生成新样本
        data, label = next(iter(dataset))
        
        # 提取IQ数据（转换为numpy）
        if hasattr(data, 'numpy'):
            clean_signal = data.numpy()
        else:
            clean_signal = np.array(data)
        
        # 归一化功率
        clean_signal = normalize_signal_power(clean_signal, target_power=1.0)
        
        # 应用RFF损伤
        rff_simulator = self.rff_simulators[device_id]
        signal_with_rff = rff_simulator.apply_all_impairments(
            clean_signal,
            center_freq=label.get('center_freq', 0.0)
        )
        
        # 归一化（RFF可能改变功率）
        signal_with_rff = normalize_signal_power(signal_with_rff, target_power=1.0)
        
        # 添加噪声
        final_signal = add_awgn(signal_with_rff, snr_db)
        
        # 构建元数据
        combined_class_id = get_combined_class_id(device_id, mod_id, self.config.NUM_MODULATIONS)
        
        metadata = {
            'device_id': device_id,
            'device_name': self.devices[device_id].device_name,
            'modulation_id': mod_id,
            'modulation_name': mod_name,
            'combined_class_id': combined_class_id,
            'snr_db': snr_db,
            'rff_params': self.devices[device_id].to_dict(),
        }
        
        return final_signal, metadata
    
    def generate_dataset(self, progress_bar: bool = True):
        """生成完整数据集"""
        
        total_samples = self.config.total_samples()
        
        # 准备存储
        if self.config.USE_HDF5:
            output_file = os.path.join(self.config.OUTPUT_DIR, "radar_rff_dataset.h5")
            hdf5_file = h5py.File(output_file, 'w')
            
            # 创建数据集
            iq_data = hdf5_file.create_dataset(
                'iq_data',
                shape=(total_samples, self.config.NUM_IQ_SAMPLES),
                dtype=np.complex64,
                compression='gzip',
                compression_opts=4,
            )
            
            # 创建标签数据集
            device_labels = hdf5_file.create_dataset(
                'device_labels', shape=(total_samples,), dtype=np.int32
            )
            modulation_labels = hdf5_file.create_dataset(
                'modulation_labels', shape=(total_samples,), dtype=np.int32
            )
            combined_labels = hdf5_file.create_dataset(
                'combined_labels', shape=(total_samples,), dtype=np.int32
            )
            snr_values = hdf5_file.create_dataset(
                'snr_db', shape=(total_samples,), dtype=np.float32
            )
        else:
            # NumPy数组
            iq_data = np.zeros((total_samples, self.config.NUM_IQ_SAMPLES), dtype=np.complex64)
            device_labels = np.zeros(total_samples, dtype=np.int32)
            modulation_labels = np.zeros(total_samples, dtype=np.int32)
            combined_labels = np.zeros(total_samples, dtype=np.int32)
            snr_values = np.zeros(total_samples, dtype=np.float32)
        
        # 生成样本
        print(f"\n开始生成 {total_samples} 个样本...")
        
        sample_idx = 0
        
        # 使用tqdm显示进度
        pbar = tqdm(total=total_samples, desc="生成数据") if progress_bar else None
        
        for device_id in range(self.config.NUM_DEVICES):
            for mod_id, mod_name in enumerate(self.config.MODULATION_LIST):
                for _ in range(self.config.SAMPLES_PER_DEVICE_MODULATION):
                    # SNR配置：固定或随机
                    if self.config.FIXED_SNR_DB is not None:
                        snr_db = self.config.FIXED_SNR_DB  # 使用固定SNR
                    else:
                        snr_db = np.random.uniform(self.config.SNR_MIN_DB, self.config.SNR_MAX_DB)  # 随机SNR
                    
                    # 生成样本
                    signal, metadata = self.generate_single_sample(
                        device_id, mod_name, snr_db
                    )
                    
                    # 存储
                    iq_data[sample_idx] = signal
                    device_labels[sample_idx] = metadata['device_id']
                    modulation_labels[sample_idx] = metadata['modulation_id']
                    combined_labels[sample_idx] = metadata['combined_class_id']
                    snr_values[sample_idx] = metadata['snr_db']
                    
                    sample_idx += 1
                    
                    if pbar:
                        pbar.update(1)
        
        if pbar:
            pbar.close()
        
        # 保存数据
        if self.config.USE_HDF5:
            # HDF5已经在上面写入，只需关闭
            hdf5_file.close()
            print(f"\n✅ 数据已保存至 HDF5 文件: {output_file}")
        else:
            # 保存为NumPy文件
            np.save(os.path.join(self.config.OUTPUT_DIR, 'iq_data.npy'), iq_data)
            np.save(os.path.join(self.config.OUTPUT_DIR, 'device_labels.npy'), device_labels)
            np.save(os.path.join(self.config.OUTPUT_DIR, 'modulation_labels.npy'), modulation_labels)
            np.save(os.path.join(self.config.OUTPUT_DIR, 'combined_labels.npy'), combined_labels)
            np.save(os.path.join(self.config.OUTPUT_DIR, 'snr_db.npy'), snr_values)
            print(f"\n✅ 数据已保存至 NumPy 文件: {self.config.OUTPUT_DIR}/")
        
        # 保存元数据和配置
        self._save_metadata()
        
        print("\n" + "="*80)
        print("数据集生成完成！")
        print("="*80)
        self._print_summary()
    
    def _save_metadata(self):
        """保存数据集元数据"""
        metadata = {
            'dataset_info': {
                'name': 'Radar RFF Dataset',
                'description': '20类雷达设备 × 26种调制方式',
                'creation_date': datetime.now().isoformat(),
                'num_devices': self.config.NUM_DEVICES,
                'num_modulations': self.config.NUM_MODULATIONS,
                'total_classes': self.config.total_classes(),
                'total_samples': self.config.total_samples(),
                'samples_per_class': self.config.SAMPLES_PER_DEVICE_MODULATION,
            },
            'signal_parameters': {
                'num_iq_samples': self.config.NUM_IQ_SAMPLES,
                'sample_rate_hz': self.config.SAMPLE_RATE,
                'snr_range_db': [self.config.SNR_MIN_DB, self.config.SNR_MAX_DB],
            },
            'modulation_list': self.config.MODULATION_LIST,
            'device_list': [device.device_name for device in self.devices],
            'rff_parameters': {
                'diverse_mode': self.config.USE_DIVERSE_RFF,
                'seed': self.config.RFF_SEED,
                'devices': [device.to_dict() for device in self.devices],
            },
        }
        
        # 保存为YAML
        metadata_file = os.path.join(self.config.OUTPUT_DIR, 'metadata.yaml')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 元数据已保存至: {metadata_file}")
        
        # 另存为类别映射文本文件
        class_map_file = os.path.join(self.config.OUTPUT_DIR, 'class_mapping.txt')
        with open(class_map_file, 'w', encoding='utf-8') as f:
            f.write("设备-调制组合类别映射表\n")
            f.write("="*80 + "\n")
            f.write(f"{'组合ID':<10} {'设备ID':<10} {'调制ID':<10} {'设备名称':<25} {'调制名称':<15}\n")
            f.write("="*80 + "\n")
            
            for device_id in range(self.config.NUM_DEVICES):
                for mod_id, mod_name in enumerate(self.config.MODULATION_LIST):
                    combined_id = get_combined_class_id(device_id, mod_id, self.config.NUM_MODULATIONS)
                    device_name = self.devices[device_id].device_name
                    f.write(f"{combined_id:<10} {device_id:<10} {mod_id:<10} "
                           f"{device_name:<25} {mod_name:<15}\n")
        
        print(f"✅ 类别映射已保存至: {class_map_file}")
    
    def _print_summary(self):
        """打印数据集摘要"""
        print(f"\n数据集统计:")
        print(f"  - 雷达设备数: {self.config.NUM_DEVICES}")
        print(f"  - 调制方式数: {self.config.NUM_MODULATIONS}")
        print(f"  - 总类别数: {self.config.total_classes()}")
        print(f"  - 每类样本数: {self.config.SAMPLES_PER_DEVICE_MODULATION}")
        print(f"  - 总样本数: {self.config.total_samples()}")
        print(f"  - 每样本IQ点数: {self.config.NUM_IQ_SAMPLES}")
        print(f"  - 采样率: {self.config.SAMPLE_RATE/1e6} MHz")
        print(f"  - SNR范围: {self.config.SNR_MIN_DB}-{self.config.SNR_MAX_DB} dB")
        print(f"  - 数据集大小: ~{self.config.estimated_size_gb():.2f} GB")
        print(f"  - 输出目录: {self.config.OUTPUT_DIR}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成雷达RFF数据集')
    parser.add_argument('--num-devices', type=int, default=20, help='雷达设备数量')
    parser.add_argument('--samples-per-class', type=int, default=1000, help='每类样本数')
    parser.add_argument('--num-iq-samples', type=int, default=4096, help='每个信号的IQ样本点数')
    parser.add_argument('--output-dir', type=str, default='./radar_rff_dataset', help='输出目录')
    parser.add_argument('--use-numpy', action='store_true', help='使用NumPy格式而非HDF5')
    parser.add_argument('--realistic-rff', action='store_true', help='使用真实分布的RFF参数（更难）')
    parser.add_argument('--num-workers', type=int, default=None, help='并行进程数')
    parser.add_argument('--seed', type=int, default=12345, help='随机种子')
    parser.add_argument('--fixed-snr', type=float, default=None, help='固定SNR值(dB)，不设置则使用随机SNR范围')
    
    args = parser.parse_args()
    
    # 更新配置
    DatasetConfig.NUM_DEVICES = args.num_devices
    DatasetConfig.SAMPLES_PER_DEVICE_MODULATION = args.samples_per_class
    DatasetConfig.NUM_IQ_SAMPLES = args.num_iq_samples
    DatasetConfig.USE_HDF5 = not args.use_numpy
    DatasetConfig.USE_DIVERSE_RFF = not args.realistic_rff
    DatasetConfig.RANDOM_SEED = args.seed
    
    # 处理SNR配置
    if args.fixed_snr is not None:
        DatasetConfig.FIXED_SNR_DB = args.fixed_snr
        # 自动修改输出目录名，添加SNR后缀
        if args.output_dir == './radar_rff_dataset':
            DatasetConfig.OUTPUT_DIR = f'./radar_rff_dataset_snr{int(args.fixed_snr)}db'
        else:
            DatasetConfig.OUTPUT_DIR = args.output_dir
    else:
        DatasetConfig.OUTPUT_DIR = args.output_dir
    
    if args.num_workers is not None:
        DatasetConfig.NUM_WORKERS = args.num_workers
    
    # 打印配置
    print("="*80)
    print("雷达RFF数据集生成器")
    print("="*80)
    print(f"配置:")
    print(f"  - 设备数: {DatasetConfig.NUM_DEVICES}")
    print(f"  - 调制数: {DatasetConfig.NUM_MODULATIONS}")
    print(f"  - 每类样本数: {DatasetConfig.SAMPLES_PER_DEVICE_MODULATION}")
    print(f"  - 总样本数: {DatasetConfig.total_samples()}")
    print(f"  - 输出格式: {'HDF5' if DatasetConfig.USE_HDF5 else 'NumPy'}")
    print(f"  - RFF模式: {'差异明显' if DatasetConfig.USE_DIVERSE_RFF else '真实分布'}")
    
    # SNR配置信息
    if DatasetConfig.FIXED_SNR_DB is not None:
        print(f"  - SNR配置: 固定 {DatasetConfig.FIXED_SNR_DB} dB")
    else:
        print(f"  - SNR配置: 随机范围 [{DatasetConfig.SNR_MIN_DB}, {DatasetConfig.SNR_MAX_DB}] dB")
    
    print(f"  - 输出目录: {DatasetConfig.OUTPUT_DIR}")
    print(f"  - 随机种子: {DatasetConfig.RANDOM_SEED}")
    print("="*80)
    
    # 创建生成器
    generator = RadarRFFDatasetGenerator(DatasetConfig)
    
    # 生成数据集
    generator.generate_dataset(progress_bar=True)
    
    print("\n🎉 全部完成!")


if __name__ == "__main__":
    main()

