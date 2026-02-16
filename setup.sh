#!/bin/bash
# EAN 系统自动部署脚本 (Linux/macOS)
# 使用方法: chmod +x setup.sh && ./setup.sh

echo "========================================"
echo "EAN 视觉系统 - 自动部署脚本"
echo "========================================"
echo ""

# 检查 Python 版本
echo "[1/6] 检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.8 或更高版本"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "找到 Python: $PYTHON_VERSION"

# 创建虚拟环境
echo ""
echo "[2/6] 创建 Python 虚拟环境..."
if [ -d ".venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败"
        exit 1
    fi
    echo "虚拟环境创建成功"
fi

# 激活虚拟环境
echo ""
echo "[3/6] 激活虚拟环境..."
source .venv/bin/activate
echo "虚拟环境已激活"

# 升级 pip
echo ""
echo "[4/6] 升级 pip..."
python -m pip install --upgrade pip --quiet
echo "pip 升级完成"

# 安装依赖
echo ""
echo "[5/6] 安装项目依赖（这可能需要几分钟）..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi
echo "依赖安装完成"

# 创建数据目录
echo ""
echo "[6/6] 创建数据目录结构..."
directories=(
    "data"
    "data/raw"
    "data/labels"
    "data/labels/masks"
    "data/labels/crops"
    "data/datasets"
    "data/models"
    "data/mappings"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "创建目录: $dir"
    fi
done
echo "数据目录创建完成"

# 完成
echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "下一步操作:"
echo "1. 将产品照片放入 data/raw/ 目录"
echo "2. 运行标注工具: python tools/label_ui_gradio.py"
echo "3. 查看完整使用指南: 部署指南.md"
echo ""
echo "启动 API 服务: ./run_api.sh"
echo "启动标注工具: ./run_labeling.sh"
echo ""
