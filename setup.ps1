# EAN 系统自动部署脚本 (Windows PowerShell)
# 使用方法: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EAN 视觉系统 - 自动部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 版本
Write-Host "[1/6] 检查 Python 版本..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 未找到 Python，请先安装 Python 3.8 或更高版本" -ForegroundColor Red
    exit 1
}
Write-Host "找到 Python: $pythonVersion" -ForegroundColor Green

# 创建虚拟环境
Write-Host ""
Write-Host "[2/6] 创建 Python 虚拟环境..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "虚拟环境已存在，跳过创建" -ForegroundColor Green
} else {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 创建虚拟环境失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "虚拟环境创建成功" -ForegroundColor Green
}

# 激活虚拟环境
Write-Host ""
Write-Host "[3/6] 激活虚拟环境..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
Write-Host "虚拟环境已激活" -ForegroundColor Green

# 升级 pip
Write-Host ""
Write-Host "[4/6] 升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "pip 升级完成" -ForegroundColor Green

# 安装依赖
Write-Host ""
Write-Host "[5/6] 安装项目依赖（这可能需要几分钟）..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 依赖安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "依赖安装完成" -ForegroundColor Green

# 创建数据目录
Write-Host ""
Write-Host "[6/6] 创建数据目录结构..." -ForegroundColor Yellow
$directories = @(
    "data",
    "data\raw",
    "data\labels",
    "data\labels\masks",
    "data\labels\crops",
    "data\datasets",
    "data\models",
    "data\mappings"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "创建目录: $dir" -ForegroundColor Gray
    }
}
Write-Host "数据目录创建完成" -ForegroundColor Green

# 完成
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "部署完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步操作:" -ForegroundColor Yellow
Write-Host "1. 将产品照片放入 data\raw\ 目录" -ForegroundColor White
Write-Host "2. 运行标注工具: python tools\label_ui_gradio.py" -ForegroundColor White
Write-Host "3. 查看完整使用指南: 部署指南.md" -ForegroundColor White
Write-Host ""
Write-Host "启动 API 服务: .\run_api.ps1" -ForegroundColor Cyan
Write-Host "启动标注工具: .\run_labeling.ps1" -ForegroundColor Cyan
Write-Host ""
