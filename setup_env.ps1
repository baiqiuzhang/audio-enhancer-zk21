# AudioEnhancer_zk21 环境安装脚本
#
# 只在本项目目录下创建独立环境 .venv，不安装、不升级、不卸载 Anaconda base 里的任何包。
# 回退方式：删除本目录下的 .venv 即可，系统无残留。
#
# 用法（在项目根目录）：
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -All          # 顺带下载全部模型（约 80 MB）
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -SkipModels   # 只装环境，不下载模型
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Python "C:\path\to\python.exe"

param(
    [string]$Python = "",
    [switch]$All,
    [switch]$SkipModels,
    [switch]$SkipChecks
)

$ErrorActionPreference = 'Stop'

# 让 PowerShell 以 UTF-8 解码子进程输出，否则 Python 的中文输出会在控制台变成乱码
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONIOENCODING = 'utf-8'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host ""
Write-Host "AudioEnhancer_zk21 — 环境安装" -ForegroundColor Cyan
Write-Host "项目目录: $root"
Write-Host ""

# --- 1. 找一个可用的 Python 3.12 ------------------------------------------
function Resolve-Python {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "指定的 Python 不存在: $Explicit" }
        return $Explicit
    }
    foreach ($cand in @('py -3.12', 'py -3', 'python')) {
        $parts = $cand.Split(' ')
        $exe = $parts[0]
        $cmdArgs = @()
        if ($parts.Length -gt 1) { $cmdArgs = $parts[1..($parts.Length - 1)] }

        $found = Get-Command $exe -ErrorAction SilentlyContinue
        if (-not $found) { continue }

        try {
            $out = & $exe @cmdArgs -c "import sys; print(sys.executable); print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -ne 0) { continue }
            $lines = @($out)
            if ($lines.Count -ge 2 -and $lines[1] -match '^3\.(1[0-9])$') {
                $ver = [int]($lines[1].Split('.')[1])
                if ($ver -ge 10) {
                    Write-Host "使用 Python: $($lines[0])  ($($lines[1]))"
                    return $lines[0].Trim()
                }
            }
        } catch { continue }
    }
    throw "找不到 Python 3.10+。请用 -Python 参数显式指定解释器路径。"
}

$py = Resolve-Python -Explicit $Python

# --- 2. 建立独立 venv -----------------------------------------------------
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    Write-Host ""
    Write-Host ".venv 已存在，跳过创建（如需重建请先手动删除 .venv 目录）" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "创建独立环境 .venv ..." -ForegroundColor Green
    & $py -m venv (Join-Path $root ".venv")
    if (-not (Test-Path $venvPy)) { throw "venv 创建失败" }
}

# --- 3. 安装依赖 ----------------------------------------------------------
Write-Host ""
Write-Host "升级 pip 并安装依赖 ..." -ForegroundColor Green
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

# --- 4. 自检 --------------------------------------------------------------
if (-not $SkipChecks) {
    Write-Host ""
    Write-Host "运行自检 ..." -ForegroundColor Green
    & $venvPy (Join-Path $root "tests\check_setup.py")
}

# --- 5. 下载模型 ----------------------------------------------------------
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "下载模型权重 ..." -ForegroundColor Green
    $dl = Join-Path $root "tests\download_models.py"
    if ($All) {
        & $venvPy $dl --all
    } else {
        & $venvPy $dl
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "部分模型下载失败。这不影响启动——应用界面里可以随时重试下载。" -ForegroundColor Yellow
        Write-Host "注意：本机对 github.com 时通时断，下载器已优先走 hf-mirror 镜像。" -ForegroundColor Yellow
    }
}

# --- 6. 完成 --------------------------------------------------------------
Write-Host ""
Write-Host "完成。" -ForegroundColor Cyan
Write-Host ""
Write-Host "启动应用："
Write-Host "    .\.venv\Scripts\python.exe run.py" -ForegroundColor White
Write-Host ""
Write-Host "如仍缺少某些模型，可在应用界面选中模型后点「下载所选模型」，或运行："
Write-Host "    .\.venv\Scripts\python.exe tests\download_models.py --all" -ForegroundColor White
Write-Host ""
