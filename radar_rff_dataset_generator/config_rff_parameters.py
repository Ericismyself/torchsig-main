"""
雷达设备射频指纹（RFF）参数配置

为20个不同的雷达设备配置独特的硬件损伤参数，模拟真实设备的射频指纹特征。

作者：根据研究需求生成
日期：2025-10-18
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RFFParameters:
    """单个雷达设备的射频指纹参数"""
    device_id: int
    device_name: str
    
    # I/Q不平衡参数
    iq_imbalance_amplitude: float      # I/Q通道幅度不平衡 (0.95-1.05)
    iq_imbalance_phase_deg: float      # I/Q通道相位不平衡 (-5° ~ +5°)
    
    # 频率偏移参数
    carrier_freq_offset_ppm: float     # 载波频率偏移 (-50ppm ~ +50ppm)
    sampling_rate_offset_ppm: float    # 采样率偏移 (-20ppm ~ +20ppm)
    
    # 相位噪声参数
    phase_noise_level: float           # 相位噪声强度 (0.001-0.01)
    
    # 功率放大器非线性参数
    pa_rapp_smoothness: float          # Rapp模型平滑因子 (1.0-5.0)
    pa_saturation_level: float         # 饱和电平 (0.7-1.2)
    
    # DC偏移
    dc_offset_i: float                 # I通道DC偏移 (-0.05 ~ +0.05)
    dc_offset_q: float                 # Q通道DC偏移 (-0.05 ~ +0.05)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'device_id': self.device_id,
            'device_name': self.device_name,
            'iq_imbalance_amplitude': self.iq_imbalance_amplitude,
            'iq_imbalance_phase_deg': self.iq_imbalance_phase_deg,
            'carrier_freq_offset_ppm': self.carrier_freq_offset_ppm,
            'sampling_rate_offset_ppm': self.sampling_rate_offset_ppm,
            'phase_noise_level': self.phase_noise_level,
            'pa_rapp_smoothness': self.pa_rapp_smoothness,
            'pa_saturation_level': self.pa_saturation_level,
            'dc_offset_i': self.dc_offset_i,
            'dc_offset_q': self.dc_offset_q,
        }


def generate_device_parameters(
    num_devices: int = 20,
    seed: int = 42,
    diverse: bool = True
) -> List[RFFParameters]:
    """
    生成所有雷达设备的RFF参数
    
    参数:
        num_devices: 设备数量 (默认20)
        seed: 随机种子，保证可复现
        diverse: 是否生成差异明显的参数（True=更容易区分，False=更真实但更难）
    
    返回:
        包含所有设备RFF参数的列表
    """
    np.random.seed(seed)
    devices = []
    
    for i in range(num_devices):
        if diverse:
            # 差异明显的参数设置（便于初期研究）
            # 将参数空间划分，确保设备间有显著差异
            
            # I/Q幅度不平衡：分成5档
            iq_amp_levels = [0.96, 0.98, 1.0, 1.02, 1.04]
            iq_amp = iq_amp_levels[i % len(iq_amp_levels)]
            
            # I/Q相位不平衡：分成4档
            iq_phase_levels = [-4, -2, 2, 4]
            iq_phase = iq_phase_levels[i % len(iq_phase_levels)]
            
            # 载波频偏：分组分配
            cfo_base = -40 + (i * 4)  # 每个设备间隔4ppm
            cfo = cfo_base + np.random.uniform(-1, 1)
            
            # 采样率偏移
            sro_base = -15 + (i * 1.5)
            sro = sro_base + np.random.uniform(-0.3, 0.3)
            
            # 相位噪声：分成几个档次
            phase_noise_levels = [0.002, 0.004, 0.006, 0.008]
            phase_noise = phase_noise_levels[i % len(phase_noise_levels)]
            
            # 功放参数
            pa_smooth = 1.5 + (i % 4) * 0.8  # 1.5, 2.3, 3.1, 3.9
            pa_sat = 0.85 + (i % 3) * 0.1    # 0.85, 0.95, 1.05
            
            # DC偏移
            dc_i = -0.04 + (i % 5) * 0.02  # -0.04, -0.02, 0, 0.02, 0.04
            dc_q = -0.03 + (i % 4) * 0.02  # -0.03, -0.01, 0.01, 0.03
            
        else:
            # 真实分布的参数设置（更具挑战性）
            # 参数在合理范围内随机采样
            
            iq_amp = np.random.uniform(0.95, 1.05)
            iq_phase = np.random.uniform(-5, 5)
            cfo = np.random.uniform(-50, 50)
            sro = np.random.uniform(-20, 20)
            phase_noise = np.random.uniform(0.001, 0.01)
            pa_smooth = np.random.uniform(1.0, 5.0)
            pa_sat = np.random.uniform(0.7, 1.2)
            dc_i = np.random.uniform(-0.05, 0.05)
            dc_q = np.random.uniform(-0.05, 0.05)
        
        device = RFFParameters(
            device_id=i,
            device_name=f"Radar_Device_{i+1:02d}",
            iq_imbalance_amplitude=iq_amp,
            iq_imbalance_phase_deg=iq_phase,
            carrier_freq_offset_ppm=cfo,
            sampling_rate_offset_ppm=sro,
            phase_noise_level=phase_noise,
            pa_rapp_smoothness=pa_smooth,
            pa_saturation_level=pa_sat,
            dc_offset_i=dc_i,
            dc_offset_q=dc_q,
        )
        devices.append(device)
    
    return devices


def print_device_summary(devices: List[RFFParameters]):
    """打印所有设备参数的摘要"""
    print("=" * 100)
    print(f"{'设备名称':<20} {'IQ幅度':<10} {'IQ相位(°)':<12} {'CFO(ppm)':<12} "
          f"{'相位噪声':<12} {'功放平滑':<10} {'DC_I':<10} {'DC_Q':<10}")
    print("=" * 100)
    
    for dev in devices:
        print(f"{dev.device_name:<20} "
              f"{dev.iq_imbalance_amplitude:<10.4f} "
              f"{dev.iq_imbalance_phase_deg:<12.2f} "
              f"{dev.carrier_freq_offset_ppm:<12.2f} "
              f"{dev.phase_noise_level:<12.4f} "
              f"{dev.pa_rapp_smoothness:<10.2f} "
              f"{dev.dc_offset_i:<10.4f} "
              f"{dev.dc_offset_q:<10.4f}")
    
    print("=" * 100)
    print(f"总计: {len(devices)} 个雷达设备")
    print("=" * 100)


# 26种调制方式列表（与TorchSig兼容）
MODULATION_LIST = [
    # QAM系列 (4种)
    "16qam", "64qam", "256qam", "1024qam",
    
    # PSK系列 (5种)
    "bpsk", "qpsk", "8psk", "16psk", "32psk",
    
    # FSK系列 (4种)
    "2fsk", "4fsk", "8fsk", "16fsk",
    
    # GFSK系列 (4种)
    "2gfsk", "4gfsk", "8gfsk", "16gfsk",
    
    # MSK系列 (4种)
    "2msk", "4msk", "8msk", "16msk",
    
    # AM系列 (4种)
    "am-dsb", "am-dsb-sc", "am-lsb", "am-usb",
    
    # FM系列 (1种)
    "fm"
]


def get_combined_class_id(device_id: int, modulation_id: int, num_modulations: int = 26) -> int:
    """
    计算设备-调制组合的唯一类别ID
    
    参数:
        device_id: 设备ID (0-19)
        modulation_id: 调制ID (0-25)
        num_modulations: 调制方式总数 (26)
    
    返回:
        组合类别ID (0-519)
    """
    return device_id * num_modulations + modulation_id


def get_device_and_modulation(combined_class_id: int, num_modulations: int = 26) -> tuple:
    """
    从组合类别ID反向获取设备ID和调制ID
    
    参数:
        combined_class_id: 组合类别ID (0-519)
        num_modulations: 调制方式总数 (26)
    
    返回:
        (device_id, modulation_id) 元组
    """
    device_id = combined_class_id // num_modulations
    modulation_id = combined_class_id % num_modulations
    return device_id, modulation_id


# ============================================================================
# 测试和演示
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("雷达设备RFF参数配置测试")
    print("="*100)
    
    # 生成20个设备的参数
    print("\n方案1: 差异明显的参数（推荐用于初期研究）")
    devices_diverse = generate_device_parameters(num_devices=20, seed=42, diverse=True)
    print_device_summary(devices_diverse)
    
    print("\n方案2: 真实分布的参数（更具挑战性）")
    devices_realistic = generate_device_parameters(num_devices=20, seed=42, diverse=False)
    print_device_summary(devices_realistic)
    
    # 显示调制类型
    print("\n" + "="*100)
    print(f"支持的调制方式（共{len(MODULATION_LIST)}种）:")
    print("="*100)
    for i, mod in enumerate(MODULATION_LIST):
        print(f"{i:2d}. {mod}")
    
    # 显示组合示例
    print("\n" + "="*100)
    print("设备-调制组合示例:")
    print("="*100)
    print(f"{'组合ID':<10} {'设备':<20} {'调制':<15}")
    print("-"*100)
    
    # 显示前10个和后10个组合
    for i in [0, 1, 2, 25, 26, 27, 51, 52, 518, 519]:
        dev_id, mod_id = get_device_and_modulation(i)
        print(f"{i:<10} {devices_diverse[dev_id].device_name:<20} {MODULATION_LIST[mod_id]:<15}")
    
    print("="*100)
    print(f"\n总计: {20 * 26} 个不同的（设备-调制）组合")
    print(f"数据集标注: 每个样本同时包含设备标签(0-19)和调制标签(0-25)")
    print("="*100)
    
    # 统计信息
    print("\n" + "="*100)
    print("参数分布统计:")
    print("="*100)
    
    iq_amps = [d.iq_imbalance_amplitude for d in devices_diverse]
    cfos = [d.carrier_freq_offset_ppm for d in devices_diverse]
    phase_noises = [d.phase_noise_level for d in devices_diverse]
    
    print(f"I/Q幅度不平衡范围: [{min(iq_amps):.4f}, {max(iq_amps):.4f}]")
    print(f"载波频偏范围: [{min(cfos):.2f}, {max(cfos):.2f}] ppm")
    print(f"相位噪声范围: [{min(phase_noises):.4f}, {max(phase_noises):.4f}]")
    print("="*100)

