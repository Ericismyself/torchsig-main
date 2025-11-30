"""
雷达射频指纹（RFF）数据集生成器

用于生成包含20类雷达设备和26种调制方式的数据集，
支持雷达辐射源识别（Specific Emitter Identification, SEI）研究。

主要模块：
- config_rff_parameters: RFF参数配置
- rff_impairments: 硬件损伤模拟
- generate_dataset: 数据集生成主脚本

作者：根据研究需求生成
日期：2025-10-18
"""

__version__ = "1.0.0"
__author__ = "Radar RFF Research Team"

from .config_rff_parameters import (
    RFFParameters,
    generate_device_parameters,
    MODULATION_LIST,
    get_combined_class_id,
    get_device_and_modulation,
)

from .rff_impairments import (
    RFFImpairmentSimulator,
    normalize_signal_power,
    add_awgn,
)

__all__ = [
    'RFFParameters',
    'generate_device_parameters',
    'MODULATION_LIST',
    'get_combined_class_id',
    'get_device_and_modulation',
    'RFFImpairmentSimulator',
    'normalize_signal_power',
    'add_awgn',
]

