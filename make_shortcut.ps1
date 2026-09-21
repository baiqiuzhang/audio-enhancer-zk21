# 在桌面创建 AudioEnhancer_zk21 的快捷方式。
#
# 可重复运行：同名快捷方式会被覆盖。
# 用 pythonw.exe 启动，因此不会弹出控制台窗口；启动失败时会弹系统消息框，
# 并把详情写入 out\startup_error.log。
#
# 用法（在项目根目录）：
#   powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
#   powershell -ExecutionPolicy Bypass -File make_shortcut.ps1 -StartMenu   # 同时放进开始菜单

param(
    [string]$Name = "AudioEnhancer_zk21",
    [switch]$StartMenu
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$runner  = Join-Path $root "run.py"
$icon    = Join-Path $root "icon.ico"

if (-not (Test-Path $pythonw)) { throw "找不到 $pythonw —— 请先运行 setup_env.ps1" }
if (-not (Test-Path $runner))  { throw "找不到 $runner —— 项目文件不完整？" }

if ($StartMenu) {
    $dir = Join-Path ([Environment]::GetFolderPath('Programs')) 'AudioEnhancer_zk21'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
} else {
    $dir = [Environment]::GetFolderPath('Desktop')
}
$lnk = Join-Path $dir "$Name.lnk"

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($lnk)
$shortcut.TargetPath       = $pythonw
$shortcut.Arguments        = '"' + $runner + '"'
$shortcut.WorkingDirectory = $root
$shortcut.Description      = 'AudioEnhancer_zk21 — 本地语音增强：导入音频/视频，降噪增强，A/B 试听'
$shortcut.WindowStyle      = 1
if (Test-Path $icon) {
    $shortcut.IconLocation = "$icon,0"     # 参数名是 IconLocation（不是 IconLocationName）
} else {
    Write-Host "提示：未找到 icon.ico，将使用默认图标。可运行 make_icon.py 生成。" -ForegroundColor Yellow
}
$shortcut.Save()

Write-Host ""
Write-Host "已创建快捷方式" -ForegroundColor Green
Write-Host "  位置：$lnk"
Write-Host "  目标：$pythonw"
Write-Host "  参数：$runner"
Write-Host "  工作目录：$root"
Write-Host "  图标：$(if (Test-Path $icon) { $icon } else { '（默认）' })"
Write-Host ""
Write-Host "双击即可启动（无控制台窗口）。若启动失败会弹消息框，详情见 out\startup_error.log。"
Write-Host ""
