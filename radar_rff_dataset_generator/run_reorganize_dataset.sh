#!/bin/bash
################################################################################
# 数据集重组工具 - 批量处理多SNR数据集
# 
# 功能：将多个SNR的大HDF5文件分别重组为两种结构
#   1. 按设备组织 (20个文件): 每个文件包含1个设备的26种调制
#   2. 按调制组织 (26个文件): 每个文件包含1种调制的20个设备
#
# 使用场景：
#   - 设备指纹识别研究 → 使用 by_device/
#   - 调制识别研究 → 使用 by_modulation/
#   - 灵活加载，节省内存
#
# 处理的数据集：
#   - radar_rff_dataset_snr5db
#   - radar_rff_dataset_snr10db
#   - radar_rff_dataset_snr15db
#   - radar_rff_dataset_snr20db
#   - radar_rff_dataset_snr25db
################################################################################

# 进入项目目录
cd "$(dirname "$0")"

# 定义SNR值列表（与生成脚本保持一致）
SNR_LIST=(5 10 15 20 25)

# 开始时间
START_TIME=$(date +%s)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🔄 开始批量重组多SNR数据集                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 配置信息："
echo "  - SNR值列表: ${SNR_LIST[@]} dB"
echo "  - 数据集数量: ${#SNR_LIST[@]} 个"
echo "  - 设备数: 20"
echo "  - 调制数: 26"
echo ""

# 计数器
SUCCESS_COUNT=0
FAIL_COUNT=0

# 循环处理每个SNR的数据集
for SNR in "${SNR_LIST[@]}"
do
    echo "════════════════════════════════════════════════════════════════════"
    echo "🎯 正在重组 SNR = ${SNR} dB 的数据集..."
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    INPUT_DIR="./radar_rff_dataset_snr${SNR}db"
    OUTPUT_DIR="./radar_rff_dataset_snr${SNR}db_organized"
    INPUT_H5="${INPUT_DIR}/radar_rff_dataset.h5"
    
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_H5" ]; then
        echo "⚠️  警告: 未找到输入文件 $INPUT_H5"
        echo "   跳过此数据集..."
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # 记录单个数据集开始时间
    DATASET_START=$(date +%s)
    
    # 重组数据集
    python organize_dataset.py \
        --input-h5 "$INPUT_H5" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-devices 20 \
        --num-modulations 26
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        DATASET_END=$(date +%s)
        DATASET_TIME=$((DATASET_END - DATASET_START))
        echo ""
        echo "✅ SNR = ${SNR} dB 数据集重组完成！"
        echo "⏱️  耗时: $((DATASET_TIME / 60)) 分钟 $((DATASET_TIME % 60)) 秒"
        echo "📁 输出目录: $OUTPUT_DIR"
        echo ""
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ SNR = ${SNR} dB 数据集重组失败！"
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# 结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ✅ 批量重组完成！                                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 重组统计："
echo "  - 成功: $SUCCESS_COUNT 个数据集"
echo "  - 失败: $FAIL_COUNT 个数据集"
echo "  - 总耗时: $((TOTAL_TIME / 3600)) 小时 $(((TOTAL_TIME % 3600) / 60)) 分钟"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "📁 重组后的输出目录："
    for SNR in "${SNR_LIST[@]}"
    do
        OUTPUT_DIR="./radar_rff_dataset_snr${SNR}db_organized"
        if [ -d "$OUTPUT_DIR" ]; then
            echo "  - $OUTPUT_DIR/"
            echo "    ├── by_device/       (20个文件，每个约1.25 GB)"
            echo "    ├── by_modulation/   (26个文件，每个约0.96 GB)"
            echo "    ├── metadata.yaml"
            echo "    ├── README.md"
            echo "    └── ..."
        fi
    done
    echo ""
fi

echo "💡 使用说明："
echo "  - 设备指纹识别研究 → 使用 by_device/ 目录"
echo "  - 调制识别研究 → 使用 by_modulation/ 目录"
echo "  - 可以只加载需要的子集，节省内存"
echo ""

