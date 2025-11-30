#!/usr/bin/env python
"""
标签查询工具 - 快速查询设备和调制标签

使用示例：
    # 列出所有设备
    python query_labels.py --list-devices
    
    # 列出所有调制
    python query_labels.py --list-modulations
    
    # 查询组合ID=100对应的设备和调制
    python query_labels.py --combined-id 100
    
    # 查询设备5、调制10对应的组合ID
    python query_labels.py --device-id 5 --modulation-id 10
"""
import yaml
import argparse
import os

def load_metadata(metadata_file):
    """加载元数据"""
    if not os.path.exists(metadata_file):
        print(f"❌ 找不到元数据文件: {metadata_file}")
        print(f"   请先生成数据集，或指定正确的元数据文件路径")
        exit(1)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def query_by_combined_id(metadata, combined_id):
    """通过组合ID查询"""
    for item in metadata['class_mapping']:
        if item['combined_id'] == combined_id:
            return item
    return None

def query_by_device_modulation(metadata, device_id, modulation_id):
    """通过设备ID和调制ID查询"""
    for item in metadata['class_mapping']:
        if item['device_id'] == device_id and item['modulation_id'] == modulation_id:
            return item
    return None

def list_all_devices(metadata):
    """列出所有设备"""
    print("\n" + "="*60)
    print("📱 所有设备列表")
    print("="*60)
    print(f"{'设备ID':<10} {'设备名称':<30}")
    print("-"*60)
    for device in metadata['devices']:
        print(f"{device['device_id']:<10} {device['device_name']:<30}")
    print(f"\n总计: {len(metadata['devices'])} 个设备")

def list_all_modulations(metadata):
    """列出所有调制"""
    print("\n" + "="*60)
    print("📡 所有调制方式")
    print("="*60)
    print(f"{'调制ID':<10} {'调制名称':<30}")
    print("-"*60)
    for mod in metadata['modulations']:
        print(f"{mod['modulation_id']:<10} {mod['modulation_name']:<30}")
    print(f"\n总计: {len(metadata['modulations'])} 种调制")

def list_device_modulations(metadata, device_id):
    """列出某个设备的所有调制组合"""
    device_name = None
    for device in metadata['devices']:
        if device['device_id'] == device_id:
            device_name = device['device_name']
            break
    
    if device_name is None:
        print(f"❌ 找不到设备ID {device_id}")
        return
    
    print("\n" + "="*80)
    print(f"📊 设备 {device_id} ({device_name}) 的所有调制组合")
    print("="*80)
    print(f"{'组合ID':<10} {'调制ID':<10} {'调制名称':<30}")
    print("-"*80)
    
    for item in metadata['class_mapping']:
        if item['device_id'] == device_id:
            print(f"{item['combined_id']:<10} {item['modulation_id']:<10} {item['modulation_name']:<30}")
    
    print(f"\n总计: {len(metadata['modulations'])} 个组合")

def list_modulation_devices(metadata, modulation_id):
    """列出某个调制的所有设备组合"""
    modulation_name = None
    for mod in metadata['modulations']:
        if mod['modulation_id'] == modulation_id:
            modulation_name = mod['modulation_name']
            break
    
    if modulation_name is None:
        print(f"❌ 找不到调制ID {modulation_id}")
        return
    
    print("\n" + "="*80)
    print(f"📊 调制 {modulation_id} ({modulation_name}) 的所有设备组合")
    print("="*80)
    print(f"{'组合ID':<10} {'设备ID':<10} {'设备名称':<30}")
    print("-"*80)
    
    for item in metadata['class_mapping']:
        if item['modulation_id'] == modulation_id:
            print(f"{item['combined_id']:<10} {item['device_id']:<10} {item['device_name']:<30}")
    
    print(f"\n总计: {len(metadata['devices'])} 个组合")

def show_dataset_summary(metadata):
    """显示数据集摘要"""
    print("\n" + "="*80)
    print("📊 数据集摘要")
    print("="*80)
    print(f"数据集名称: {metadata.get('dataset_name', 'N/A')}")
    print(f"创建日期: {metadata.get('creation_date', 'N/A')}")
    print(f"RFF模式: {metadata.get('rff_mode', 'N/A')}")
    print()
    print(f"设备数量: {metadata['num_devices']}")
    print(f"调制数量: {metadata['num_modulations']}")
    print(f"每类样本数: {metadata['samples_per_class']}")
    print(f"总类别数: {metadata['total_classes']}")
    print(f"总样本数: {metadata['total_samples']:,}")
    print()
    print(f"IQ采样点数: {metadata['num_iq_samples']}")
    print(f"采样率: {metadata['sample_rate']} Hz")
    print(f"SNR范围: {metadata['snr_range_db'][0]} ~ {metadata['snr_range_db'][1]} dB")

def decode_combined_id(combined_id, num_modulations):
    """解码组合ID"""
    device_id = combined_id // num_modulations
    modulation_id = combined_id % num_modulations
    return device_id, modulation_id

def main():
    parser = argparse.ArgumentParser(
        description='标签查询工具 - 查询雷达RFF数据集的标签信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示数据集摘要
  python query_labels.py --summary
  
  # 列出所有设备
  python query_labels.py --list-devices
  
  # 列出所有调制
  python query_labels.py --list-modulations
  
  # 查询组合ID=100对应的设备和调制
  python query_labels.py --combined-id 100
  
  # 查询设备5、调制10对应的组合ID
  python query_labels.py --device-id 5 --modulation-id 10
  
  # 列出设备3的所有调制组合
  python query_labels.py --list-device-mods 3
  
  # 列出调制5的所有设备组合
  python query_labels.py --list-mod-devices 5
        """
    )
    
    parser.add_argument('--metadata', default='./radar_rff_dataset_realistic/metadata.yaml',
                       help='元数据文件路径 (默认: ./radar_rff_dataset_realistic/metadata.yaml)')
    parser.add_argument('--summary', action='store_true', help='显示数据集摘要')
    parser.add_argument('--list-devices', action='store_true', help='列出所有设备')
    parser.add_argument('--list-modulations', action='store_true', help='列出所有调制')
    parser.add_argument('--combined-id', type=int, help='查询组合ID (0-519)')
    parser.add_argument('--device-id', type=int, help='设备ID (0-19)')
    parser.add_argument('--modulation-id', type=int, help='调制ID (0-25)')
    parser.add_argument('--list-device-mods', type=int, metavar='DEVICE_ID',
                       help='列出指定设备的所有调制组合')
    parser.add_argument('--list-mod-devices', type=int, metavar='MODULATION_ID',
                       help='列出指定调制的所有设备组合')
    
    args = parser.parse_args()
    
    # 加载元数据
    metadata = load_metadata(args.metadata)
    
    # 显示数据集摘要
    if args.summary:
        show_dataset_summary(metadata)
        return
    
    # 列出所有设备
    if args.list_devices:
        list_all_devices(metadata)
        return
    
    # 列出所有调制
    if args.list_modulations:
        list_all_modulations(metadata)
        return
    
    # 列出设备的所有调制组合
    if args.list_device_mods is not None:
        list_device_modulations(metadata, args.list_device_mods)
        return
    
    # 列出调制的所有设备组合
    if args.list_mod_devices is not None:
        list_modulation_devices(metadata, args.list_mod_devices)
        return
    
    # 通过组合ID查询
    if args.combined_id is not None:
        result = query_by_combined_id(metadata, args.combined_id)
        if result:
            print("\n" + "="*60)
            print(f"🔍 组合ID {args.combined_id} 的信息")
            print("="*60)
            print(f"设备ID:   {result['device_id']}")
            print(f"设备名称: {result['device_name']}")
            print(f"调制ID:   {result['modulation_id']}")
            print(f"调制名称: {result['modulation_name']}")
            print()
            print(f"💡 解码公式: 设备ID={result['combined_id']} // 26 = {result['device_id']}")
            print(f"           调制ID={result['combined_id']} % 26 = {result['modulation_id']}")
        else:
            print(f"❌ 未找到组合ID {args.combined_id}")
            print(f"   有效范围: 0 ~ {metadata['total_classes']-1}")
        return
    
    # 通过设备ID和调制ID查询
    if args.device_id is not None and args.modulation_id is not None:
        result = query_by_device_modulation(metadata, args.device_id, args.modulation_id)
        if result:
            print("\n" + "="*60)
            print(f"🔍 设备ID {args.device_id} + 调制ID {args.modulation_id} 的信息")
            print("="*60)
            print(f"组合ID:   {result['combined_id']}")
            print(f"设备名称: {result['device_name']}")
            print(f"调制名称: {result['modulation_name']}")
            print()
            print(f"💡 编码公式: 组合ID = {args.device_id} * 26 + {args.modulation_id} = {result['combined_id']}")
        else:
            print(f"❌ 未找到对应的组合")
            print(f"   设备ID有效范围: 0 ~ {metadata['num_devices']-1}")
            print(f"   调制ID有效范围: 0 ~ {metadata['num_modulations']-1}")
        return
    
    # 如果只提供了设备ID，列出该设备的所有调制
    if args.device_id is not None:
        list_device_modulations(metadata, args.device_id)
        return
    
    # 如果只提供了调制ID，列出该调制的所有设备
    if args.modulation_id is not None:
        list_modulation_devices(metadata, args.modulation_id)
        return
    
    # 如果没有指定任何查询，显示摘要
    show_dataset_summary(metadata)
    print("\n💡 使用 --help 查看更多选项")

if __name__ == '__main__':
    main()

