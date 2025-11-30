#!/bin/bash
################################################################################
# 雷达RFF数据集生成 - 多SNR固定值模式
# 
# 配置说明：
#   - 20类雷达设备
#   - 26种调制方式
#   - 每个（设备-调制）组合生成 3000 个样本
#   - 每个信号包含 1,048,576 个IQ采样点 (TorchSig宽带默认)
#   - 采样率: 100 MHz (TorchSig宽带默认)
#   - 信号带宽: 5-10 MHz (TorchSig宽带默认)
#   - 使用HDF5格式存储（压缩）
#   - 使用真实RFF参数分布（更贴近实际）
#   - 生成11个固定SNR数据集：0dB, 2dB, 4dB, 6dB, 8dB, 10dB, 12dB, 14dB, 16dB, 18dB, 20dB
#
# 数据集规模（每个SNR）：
#   - 总类别数：20 × 26 = 520 类
#   - 总样本数：520 × 3000 = 1,560,000 个样本
#   - 估计大小：约 6.4 TB (HDF5压缩后，由于IQ采样点数增加256倍)
#
# 总体规模（11个SNR）：
#   - 总数据集：11 个
#   - 总样本数：1,560,000 × 11 = 17,160,000 个样本
#   - 总大小：约 70 TB (⚠️ 极大数据集！)
#
# 生成时间预估：
#   - 每个数据集：约 2-6 小时
#   - 总时间：约 22-66 小时
################################################################################

# 激活conda环境（如果需要）
# conda activate torchsig

# 进入项目目录
cd "$(dirname "$0")"

# 设置环境变量（可选，提高性能）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 定义要生成的SNR值列表
SNR_LIST=(0 2 4 6 8 10 12 14 16 18 20)

# 开始时间
START_TIME=$(date +%s)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🚀 开始生成多SNR雷达RFF数据集                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 配置信息："
echo "  - SNR值列表: ${SNR_LIST[@]} dB"
echo "  - 数据集数量: ${#SNR_LIST[@]} 个"
echo "  - 每个数据集: 1,560,000 样本 (~6.4 TB)"
echo "  - 设备数: 20"
echo "  - 调制数: 26"
echo "  - 每类样本数: 3000"
echo "  - IQ采样点: 1,048,576 (TorchSig宽带默认)"
echo "  - 采样率: 100 MHz"
echo "  - 信号带宽: 5-10 MHz"
echo ""

# 循环生成每个SNR的数据集
for SNR in "${SNR_LIST[@]}"
do
    echo "════════════════════════════════════════════════════════════════════"
    echo "🎯 正在生成 SNR = ${SNR} dB 的数据集..."
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    # 记录单个数据集开始时间
    DATASET_START=$(date +%s)
    
    # 生成数据集
    python generate_dataset.py \
        --num-devices 20 \
        --samples-per-class 3000 \
        --num-iq-samples 1048576 \
        --output-dir ./radar_rff_dataset_snr${SNR}db \
        --realistic-rff \
        --num-workers 8 \
        --seed 42 \
        --fixed-snr ${SNR}
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        DATASET_END=$(date +%s)
        DATASET_TIME=$((DATASET_END - DATASET_START))
        echo ""
        echo "✅ SNR = ${SNR} dB 数据集生成完成！"
        echo "⏱️  耗时: $((DATASET_TIME / 60)) 分钟 $((DATASET_TIME % 60)) 秒"
        echo "📁 输出目录: ./radar_rff_dataset_snr${SNR}db"
        echo ""
    else
        echo ""
        echo "❌ SNR = ${SNR} dB 数据集生成失败！"
        echo ""
        exit 1
    fi
done

# 结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ✅ 所有数据集生成完成！                                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 生成统计："
echo "  - 生成的数据集数量: ${#SNR_LIST[@]} 个"
echo "  - SNR列表: ${SNR_LIST[@]} dB"
echo "  - 总样本数: 17,160,000 个"
echo "  - 总耗时: $((TOTAL_TIME / 3600)) 小时 $(((TOTAL_TIME % 3600) / 60)) 分钟"
echo ""
echo "📁 输出目录："
for SNR in "${SNR_LIST[@]}"
do
    echo "  - ./radar_rff_dataset_snr${SNR}db/"
done
echo ""
echo "📄 每个数据集包含的文件："
echo "  - radar_rff_dataset.h5       # 主数据文件（HDF5格式，~6.4TB）"
echo "  - metadata.yaml              # 元数据信息"
echo "  - device_rff_parameters.yaml # 各设备的RFF参数"
echo "  - class_mapping.yaml         # 类别映射关系"
echo ""
echo "💡 下一步："
echo "  运行 ./run_reorganize_dataset.sh 对这些数据集进行重组"
echo ""

