$ErrorActionPreference = 'Stop'
$cpuStudyPath = 'C:\Users\Hoang Anh\.codex\visualizations\2026\10\03\01a10133-0b7a-7873-a4da-69e8e881f33b\cpu_study'
$cpuDataPath = 'D:\Lab_VinUni\Lab_Phase2\Lab_2\data'
New-Item -ItemType Directory -Path $cpuStudyPath -Force | Out-Null
$cpuPidFile = Join-Path $cpuStudyPath 'worker.pid'
if (Test-Path -LiteralPath $cpuPidFile) {
    $cpuExistingId = [int](Get-Content -LiteralPath $cpuPidFile)
    $cpuExisting = Get-Process -Id $cpuExistingId -ErrorAction SilentlyContinue
    if ($cpuExisting -and $cpuExisting.ProcessName -eq 'python') {
        Write-Output "CPU worker already exists: $cpuExistingId. Check cpu_status.json before starting another worker."
        exit 0
    }
}
# The exact ImageNet-1k weights have already been cached. Do not change the tag.
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$cpuTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($cpuLogName in @('training_stdout.log','training_stderr.log')) {
    $cpuLogPath = Join-Path $cpuStudyPath $cpuLogName
    if (Test-Path -LiteralPath $cpuLogPath) {
        Copy-Item -LiteralPath $cpuLogPath -Destination (Join-Path $cpuStudyPath "$cpuTimestamp-$cpuLogName")
    }
}
$cpuJob = Start-Process -FilePath (Get-Command python).Source -ArgumentList @(
    '-u', 'run_local_cpu.py', '--root', ('"' + $cpuStudyPath + '"'),
    '--data', ('"' + $cpuDataPath + '"')
) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (
    Join-Path $cpuStudyPath 'training_stdout.log'
) -RedirectStandardError (Join-Path $cpuStudyPath 'training_stderr.log') -PassThru
$cpuJob.Id | Set-Content -LiteralPath $cpuPidFile
Write-Output "Started CPU worker $($cpuJob.Id). Progress: $cpuStudyPath\cpu_status.json"
