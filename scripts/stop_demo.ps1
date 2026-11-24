$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateFile = Join-Path $ProjectRoot ".demo-runtime\server.json"

function Write-Notice {
    param([string]$Message)
    Write-Host "[ICMA] $Message" -ForegroundColor Cyan
}

if (-not (Test-Path -LiteralPath $StateFile -PathType Leaf)) {
    Write-Notice "No launcher-managed ICMA Demo server is recorded."
    exit 0
}

try {
    $state = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    $serverPid = [int]$state.pid
    $process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item -LiteralPath $StateFile -Force
        Write-Notice "The recorded server has already stopped."
        exit 0
    }

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction SilentlyContinue
    if (-not $processInfo -or $processInfo.CommandLine -notmatch "demo_server\.app") {
        throw "PID $serverPid does not appear to belong to the ICMA Demo. It was not stopped."
    }

    $taskkill = Get-Command taskkill.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $taskkill) {
        throw "taskkill.exe is unavailable."
    }

    Write-Notice "Stopping ICMA Demo process tree (PID $serverPid)"
    & $taskkill.Source /PID $serverPid /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Windows could not stop the ICMA Demo process tree."
    }

    Remove-Item -LiteralPath $StateFile -Force
    Write-Host "[OK]   ICMA Demo stopped." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
