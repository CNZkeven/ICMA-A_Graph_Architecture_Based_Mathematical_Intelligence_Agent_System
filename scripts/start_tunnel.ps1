[CmdletBinding()]
param(
    [ValidateRange(0, 65535)]
    [int]$Port = 0,
    [ValidateRange(1, 10)]
    [int]$MaxAttempts = 3,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $ProjectRoot ".demo-runtime"
$LogDir = Join-Path $ProjectRoot "logs"
$StateFile = Join-Path $RuntimeDir "tunnel.json"
$TunnelProcess = $null

function Show-PublicUrl {
    param([string]$Url)
    Write-Host ""
    Write-Host "[OK] Public demo URL: $Url" -ForegroundColor Green
    Write-Host "Keep this computer awake and connected. Run stop_tunnel.ps1 to stop sharing."
    Write-Host "Visitors share the demo's conversation history. Share only with trusted people."
    if (-not $NoBrowser) {
        try { Start-Process -FilePath $Url }
        catch { Write-Warning "Open the URL manually; the tunnel is still running." }
    }
}

try {
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    $serverFile = Join-Path $RuntimeDir "server.json"
    if ($Port -eq 0 -and (Test-Path -LiteralPath $serverFile)) {
        $serverState = Get-Content -LiteralPath $serverFile -Raw | ConvertFrom-Json
        $Port = [int]$serverState.port
    }
    if ($Port -eq 0) { $Port = 8765 }
    $target = "http://127.0.0.1:$Port"
    $health = Invoke-RestMethod -Uri "$target/api/health" -TimeoutSec 5
    if (-not $health.ok -or $health.service -ne "icma-demo") {
        throw "Start the local ICMA Demo first."
    }
    $schema = Invoke-RestMethod -Uri "$target/openapi.json" -TimeoutSec 5
    if ($schema.paths.PSObject.Properties.Name -notcontains "/api/runs/{run_id}/event-batch") {
        throw "Restart the local Demo to load the public streaming compatibility update."
    }

    if (Test-Path -LiteralPath $StateFile) {
        $known = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
        $knownProcess = Get-Process -Id ([int]$known.pid) -ErrorAction SilentlyContinue
        if ($knownProcess -and $knownProcess.ProcessName -eq "cloudflared" -and
            $knownProcess.StartTime.ToUniversalTime().Ticks.ToString() -eq $known.startedTicks) {
            if ($known.target -ne $target) {
                throw "An existing tunnel uses another port. Stop it with stop_tunnel.ps1 first."
            }
            if (-not $known.url) {
                throw "A tunnel is already starting. Check its log or stop it before retrying."
            }
            Show-PublicUrl -Url ([string]$known.url)
            exit 0
        }
        Remove-Item -LiteralPath $StateFile -Force
    }

    $cloudflared = Get-Command cloudflared.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cloudflared) { $executable = $cloudflared.Source }
    else {
        $executable = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
    }
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "cloudflared was not found. Install Cloudflare.cloudflared with winget first."
    }

    $stdoutLog = Join-Path $LogDir "cloudflared.out.log"
    $stderrLog = Join-Path $LogDir "cloudflared.err.log"
    $ready = $false
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Host "[ICMA] Creating a temporary public URL for $target (attempt $attempt/$MaxAttempts) ..." -ForegroundColor Cyan
        Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue
        $TunnelProcess = Start-Process -FilePath $executable -ArgumentList @(
            "tunnel", "--no-autoupdate", "--protocol", "http2", "--edge-ip-version", "4", "--url", $target
        ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
        $state = [PSCustomObject]@{
            pid = $TunnelProcess.Id
            executable = $executable
            startedTicks = $TunnelProcess.StartTime.ToUniversalTime().Ticks.ToString()
            target = $target
            url = ""
        }
        $state | ConvertTo-Json | Set-Content -LiteralPath $StateFile -Encoding UTF8
        for ($tick = 0; $tick -lt 90; $tick++) {
            Start-Sleep -Milliseconds 500
            $TunnelProcess.Refresh()
            $logText = Get-Content -LiteralPath $stderrLog -Raw -ErrorAction SilentlyContinue
            if ($logText -match "https://[a-z0-9-]+\.trycloudflare\.com") {
                $state.url = $Matches[0]
            }
            if ($state.url -and $logText -match "Registered tunnel connection") {
                $ready = $true
                break
            }
            if ($logText -match "failed to request quick Tunnel") { break }
            if ($TunnelProcess.HasExited) { break }
        }
        if ($ready) { break }
        # The quick-tunnel request can be reset by the network path to Cloudflare; retry with a fresh process.
        if (-not $TunnelProcess.HasExited) {
            try { $TunnelProcess.Kill(); $TunnelProcess.WaitForExit(5000) | Out-Null } catch { }
        }
        Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
        if ($attempt -lt $MaxAttempts) {
            Write-Warning "cloudflared could not create the quick tunnel (attempt $attempt/$MaxAttempts). Retrying..."
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
    if (-not $ready) {
        $tail = (Get-Content -LiteralPath $stderrLog -Tail 12 -ErrorAction SilentlyContinue) -join "`n"
        throw "The tunnel did not connect after $MaxAttempts attempt(s). Check the network and $stderrLog`n$tail"
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $StateFile -Encoding UTF8
    try {
        $publicHealth = Invoke-RestMethod -Uri "$($state.url)/api/health" -TimeoutSec 8
        if (-not $publicHealth.ok -or $publicHealth.service -ne "icma-demo") {
            Write-Warning "The tunnel connected, but its public health check did not confirm ICMA."
        }
    }
    catch { Write-Warning "The tunnel connected. Public reachability could not be verified from this network." }
    Show-PublicUrl -Url $state.url
}
catch {
    if ($TunnelProcess -and -not $TunnelProcess.HasExited) { $TunnelProcess.Kill() }
    if ($TunnelProcess) { Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue }
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
