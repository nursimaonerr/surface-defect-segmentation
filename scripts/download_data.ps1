# Severstal Steel Defect Detection veri setini Kaggle'dan indirir.
#
# On kosul: %USERPROFILE%\.kaggle\kaggle.json (Kaggle hesap ayarlarindan API token) mevcut olmali
# ve yarismanin katilim kurallari kabul edilmis olmali (kaggle.com/c/severstal-steel-defect-detection/rules).
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root "data\raw"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$kaggleExe = Join-Path $root ".venv\Scripts\kaggle.exe"
if (-not (Test-Path $kaggleExe)) { $kaggleExe = "kaggle" }
& $kaggleExe competitions download -c severstal-steel-defect-detection -p $dataDir

$zipPath = Join-Path $dataDir "severstal-steel-defect-detection.zip"
Expand-Archive -Path $zipPath -DestinationPath $dataDir -Force
Remove-Item $zipPath

Write-Output "Veri hazir: $dataDir"
Get-ChildItem $dataDir
