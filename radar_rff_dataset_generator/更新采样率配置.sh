#!/bin/bash
################################################################################
# 采样率配置更新工具
# 
# 功能：快速将采样率从 1 MHz 更新为推荐的配置
# 
# 使用方法：
#   ./更新采样率配置.sh [方案编号]
#   
# 方案选项：
#   1 - 保守方案（10 MHz）
#   2 - 均衡方案（20 MHz，推荐⭐）
#   3 - 高保真方案（100 MHz，对标TorchSig）
################################################################################

cd "$(dirname "$0")"

CONFIG_FILE="generate_dataset.py"

# 检查文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误：找不到 $CONFIG_FILE"
    exit 1
fi

# 创建备份
BACKUP_FILE="${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo "✅ 已创建备份：$BACKUP_FILE"
echo ""

# 获取方案选项
SCHEME=${1:-2}  # 默认使用方案2

case $SCHEME in
    1)
        echo "🔧 应用方案1：保守方案（10 MHz）"
        SAMPLE_RATE="10e6"
        DURATION_MIN="0.0001"
        DURATION_MAX="0.002"
        BW_MIN="5e5"
        BW_MAX="4e6"
        CF_MIN="-5e6"
        CF_MAX="5e6"
        ;;
    2)
        echo "🔧 应用方案2：均衡方案（20 MHz）⭐ 推荐"
        SAMPLE_RATE="20e6"
        DURATION_MIN="0.00005"
        DURATION_MAX="0.001"
        BW_MIN="1e6"
        BW_MAX="8e6"
        CF_MIN="-10e6"
        CF_MAX="10e6"
        ;;
    3)
        echo "🔧 应用方案3：高保真方案（100 MHz）"
        SAMPLE_RATE="100e6"
        DURATION_MIN="1e-6"
        DURATION_MAX="10e-6"
        BW_MIN="1e5"
        BW_MAX="10e6"
        CF_MIN="-50e6"
        CF_MAX="50e6"
        ;;
    *)
        echo "❌ 错误：无效的方案编号 $SCHEME"
        echo "请使用：1（保守）、2（推荐⭐）或 3（高保真）"
        exit 1
        ;;
esac

echo ""
echo "📋 配置参数："
echo "  - 采样率：$SAMPLE_RATE Hz"
echo "  - 信号时长：$DURATION_MIN - $DURATION_MAX 秒"
echo "  - 信号带宽：$BW_MIN - $BW_MAX Hz"
echo "  - 中心频率：$CF_MIN ~ $CF_MAX Hz"
echo ""

# 使用 sed 进行替换（macOS 兼容版本）
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/SAMPLE_RATE = [0-9.e+-]*/SAMPLE_RATE = $SAMPLE_RATE/" "$CONFIG_FILE"
    sed -i '' "s/SIGNAL_DURATION_MIN = [0-9.e+-]*/SIGNAL_DURATION_MIN = $DURATION_MIN/" "$CONFIG_FILE"
    sed -i '' "s/SIGNAL_DURATION_MAX = [0-9.e+-]*/SIGNAL_DURATION_MAX = $DURATION_MAX/" "$CONFIG_FILE"
    sed -i '' "s/SIGNAL_BANDWIDTH_MIN = [0-9.e+-]*/SIGNAL_BANDWIDTH_MIN = $BW_MIN/" "$CONFIG_FILE"
    sed -i '' "s/SIGNAL_BANDWIDTH_MAX = [0-9.e+-]*/SIGNAL_BANDWIDTH_MAX = $BW_MAX/" "$CONFIG_FILE"
    sed -i '' "s/SIGNAL_CENTER_FREQ_MIN = -[0-9.e+-]*/SIGNAL_CENTER_FREQ_MIN = $CF_MIN/" "$CONFIG_FILE"
    sed -i '' "s/SIGNAL_CENTER_FREQ_MAX = [0-9.e+-]*/SIGNAL_CENTER_FREQ_MAX = $CF_MAX/" "$CONFIG_FILE"
else
    # Linux
    sed -i "s/SAMPLE_RATE = [0-9.e+-]*/SAMPLE_RATE = $SAMPLE_RATE/" "$CONFIG_FILE"
    sed -i "s/SIGNAL_DURATION_MIN = [0-9.e+-]*/SIGNAL_DURATION_MIN = $DURATION_MIN/" "$CONFIG_FILE"
    sed -i "s/SIGNAL_DURATION_MAX = [0-9.e+-]*/SIGNAL_DURATION_MAX = $DURATION_MAX/" "$CONFIG_FILE"
    sed -i "s/SIGNAL_BANDWIDTH_MIN = [0-9.e+-]*/SIGNAL_BANDWIDTH_MIN = $BW_MIN/" "$CONFIG_FILE"
    sed -i "s/SIGNAL_BANDWIDTH_MAX = [0-9.e+-]*/SIGNAL_BANDWIDTH_MAX = $BW_MAX/" "$CONFIG_FILE"
    sed -i "s/SIGNAL_CENTER_FREQ_MIN = -[0-9.e+-]*/SIGNAL_CENTER_FREQ_MIN = $CF_MIN/" "$CONFIG_FILE"
    sed -i "s/SIGNAL_CENTER_FREQ_MAX = [0-9.e+-]*/SIGNAL_CENTER_FREQ_MAX = $CF_MAX/" "$CONFIG_FILE"
fi

echo "✅ 配置已更新！"
echo ""
echo "📄 查看修改："
echo "----------------------------------------"
grep -A 0 "SAMPLE_RATE = " "$CONFIG_FILE" | head -1
grep -A 0 "SIGNAL_DURATION_MIN = " "$CONFIG_FILE" | head -1
grep -A 0 "SIGNAL_DURATION_MAX = " "$CONFIG_FILE" | head -1
grep -A 0 "SIGNAL_BANDWIDTH_MIN = " "$CONFIG_FILE" | head -1
grep -A 0 "SIGNAL_BANDWIDTH_MAX = " "$CONFIG_FILE" | head -1
grep -A 0 "SIGNAL_CENTER_FREQ_MIN = " "$CONFIG_FILE" | head -1
grep -A 0 "SIGNAL_CENTER_FREQ_MAX = " "$CONFIG_FILE" | head -1
echo "----------------------------------------"
echo ""

echo "🚀 后续步骤："
echo "  1. 检查配置：cat $CONFIG_FILE | grep -A 15 'class DatasetConfig'"
echo "  2. 生成数据集：./run_generate_realistic_dataset.sh"
echo ""
echo "💡 提示：如需恢复原配置，使用备份文件："
echo "     cp $BACKUP_FILE $CONFIG_FILE"
echo ""


