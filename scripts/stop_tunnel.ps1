$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateFile = Join-Path $ProjectRoot ".demo-runtime\tunnel.json"

try {
    if (-not (Test-Path -LiteralPath $StateFile -PathType Leaf)) {
        Write-Host "[ICMA] No managed public tunnel is recorded."
        exit 0
    }
    $state = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    $tunnelProcess = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($tunnelProcess) {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$state.pid)"
        if ($tunnelProcess.ProcessName -ne "cloudflared" -or
            $tunnelProcess.StartTime.ToUniversalTime().Ticks.ToString() -ne $state.startedTicks -or
            $processInfo.ExecutablePath -ne $state.executable -or
            $processInfo.CommandLine -notmatch "--url\s+http://127\.0\.0\.1:\d+") {
            throw "The recorded PID does not match this tunnel. No process was stopped."
        }
        Stop-Process -Id $tunnelProcess.Id -Force -ErrorAction Stop
    }
    Remove-Item -LiteralPath $StateFile -Force
    Write-Host "[OK] Public tunnel stopped. The local Demo remains available." -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
