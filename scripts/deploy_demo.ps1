[CmdletBinding()]
param(
    [int]$Port = 0,
    [switch]$NoBrowser,
    [switch]$ForceInstall,
    [switch]$Rebuild,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $ProjectRoot ".demo-runtime"
$LogDir = Join-Path $ProjectRoot "logs"
$StateFile = Join-Path $RuntimeDir "server.json"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$WebRoot = Join-Path $ProjectRoot "web"
$WebDist = Join-Path $WebRoot "dist"
$Npm = $null
$ServerProcess = $null

function Write-Step {
    param([string]$Message)
    Write-Host "[ICMA] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK]   $Message" -ForegroundColor Green
}

function Write-Notice {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor DarkGray
}

function Get-TextSha256 {
    param([string]$Text)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Test-PythonCandidate {
    param(
        [string]$FilePath,
        [string[]]$Prefix = @(),
        [string]$Label = ""
    )

    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        return $null
    }

    $allPrefix = @($Prefix)
    try {
        $version = (& $FilePath @allPrefix -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null)
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        $supported = (& $FilePath @allPrefix -c "import sys; print('yes' if sys.version_info >= (3, 10) else 'no')" 2>$null)
        if ($LASTEXITCODE -ne 0 -or (($supported | Out-String).Trim() -ne "yes")) {
            return $null
        }
        return [PSCustomObject]@{
            FilePath = (Resolve-Path -LiteralPath $FilePath).Path
            Prefix = $allPrefix
            Version = (($version | Out-String).Trim())
            Label = $Label
        }
    }
    catch {
        return $null
    }
}

function Get-CondaCommand {
    $candidates = @()
    if ($env:CONDA_EXE) {
        $candidates += $env:CONDA_EXE
    }

    foreach ($name in @("conda.exe", "conda.bat", "conda")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            $candidates += $command.Source
        }
    }

    if ($env:USERPROFILE) {
        $candidates += @(
            (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
            (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
            (Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"),
            (Join-Path $env:USERPROFILE "mambaforge\Scripts\conda.exe")
        )
    }

    return $candidates | Where-Object {
        $_ -and (Test-Path -LiteralPath $_ -PathType Leaf)
    } | Select-Object -Unique
}

function Get-CompetitionPython {
    $currentPrefix = $env:CONDA_PREFIX
    if ($currentPrefix -and ((Split-Path -Leaf $currentPrefix) -ieq "competition")) {
        $python = Join-Path $currentPrefix "python.exe"
        $candidate = Test-PythonCandidate -FilePath $python -Label "conda competition"
        if ($candidate) {
            return $candidate
        }
    }

    foreach ($conda in @(Get-CondaCommand)) {
        try {
            $raw = (& $conda env list --json 2>$null)
            if ($LASTEXITCODE -ne 0) {
                continue
            }
            $data = (($raw | Out-String) | ConvertFrom-Json)
            foreach ($prefix in @($data.envs)) {
                if ((Split-Path -Leaf $prefix) -ine "competition") {
                    continue
                }
                $python = Join-Path $prefix "python.exe"
                $candidate = Test-PythonCandidate -FilePath $python -Label "conda competition"
                if ($candidate) {
                    return $candidate
                }
            }
        }
        catch {
            continue
        }
    }

    return $null
}

function Get-SystemPython {
    $candidates = @()

    foreach ($name in @("python.exe", "python")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            $candidates += [PSCustomObject]@{ FilePath = $command.Source; Prefix = @() }
        }
    }

    foreach ($name in @("py.exe", "py")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            $candidates += [PSCustomObject]@{ FilePath = $command.Source; Prefix = @("-3") }
        }
    }

    foreach ($item in $candidates) {
        $candidate = Test-PythonCandidate -FilePath $item.FilePath -Prefix $item.Prefix -Label "system Python"
        if ($candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-PythonBinary {
    param([string]$ExplicitPath)

    if ($ExplicitPath) {
        $candidate = Test-PythonCandidate -FilePath $ExplicitPath -Label "requested Python"
        if (-not $candidate) {
            throw "The requested Python is unavailable or older than 3.10: $ExplicitPath"
        }
        return $candidate
    }

    if ($env:ICMA_PYTHON) {
        $candidate = Test-PythonCandidate -FilePath $env:ICMA_PYTHON -Label "ICMA_PYTHON"
        if (-not $candidate) {
            throw "ICMA_PYTHON is unavailable or older than 3.10: $env:ICMA_PYTHON"
        }
        return $candidate
    }

    $competition = Get-CompetitionPython
    if ($competition) {
        return $competition
    }

    Write-Notice "Conda environment 'competition' was not found. Setting up a project-local .venv from system Python."
    $systemPython = Get-SystemPython
    if (-not $systemPython) {
        throw "No usable Python 3.10+ was found. Install Python 3.10 or newer, then run this launcher again."
    }

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        Write-Step "Creating project-local Python environment"
        $venvArgs = @($systemPython.Prefix) + @("-m", "venv", (Join-Path $ProjectRoot ".venv"))
        & $systemPython.FilePath @venvArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create .venv with $($systemPython.FilePath)."
        }
    }

    $candidate = Test-PythonCandidate -FilePath $venvPython -Label "project .venv"
    if (-not $candidate) {
        throw "The project-local .venv could not be initialized with Python 3.10+."
    }
    return $candidate
}

function Invoke-Python {
    param(
        [PSCustomObject]$Python,
        [string[]]$Arguments
    )

    $allArgs = @($Python.Prefix) + @($Arguments)
    & $Python.FilePath @allArgs
}

function Test-PythonImports {
    param([PSCustomObject]$Python)

    $checkFile = Join-Path $RuntimeDir "python-import-check.py"
    $check = @'
import importlib.util

modules = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "requests",
    "langgraph",
    "langchain_core",
    "chromadb",
    "sentence_transformers",
    "transformers",
    "modelscope",
    "torch",
    "sympy",
    "mpmath",
    "numpy",
    "scipy",
    "networkx",
    "sklearn",
    "Levenshtein",
    "mcp",
    "pythonjsonlogger",
]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python modules: " + ", ".join(missing))
    raise SystemExit(1)
'@

    Set-Content -LiteralPath $checkFile -Value $check -Encoding UTF8
    Invoke-Python -Python $Python -Arguments @($checkFile)
    return ($LASTEXITCODE -eq 0)
}

function Ensure-PythonDependencies {
    param([PSCustomObject]$Python)

    $requirementsText = (Get-Content -LiteralPath $RequirementsFile -Raw)
    $context = $Python.FilePath + "`n" + $Python.Version + "`n" + $requirementsText
    $stampName = "python-deps-" + (Get-TextSha256 -Text $context) + ".ok"
    $stampPath = Join-Path $RuntimeDir $stampName
    $otherStamps = @(Get-ChildItem -LiteralPath $RuntimeDir -Filter "python-deps-*.ok" -File -ErrorAction SilentlyContinue)

    if ((-not $ForceInstall) -and (Test-Path -LiteralPath $stampPath -PathType Leaf)) {
        if (Test-PythonImports -Python $Python) {
            Write-Notice "Python dependencies are ready."
            return
        }
        Remove-Item -LiteralPath $stampPath -Force
        Write-Warning "A tracked Python dependency is missing; dependency installation will run again."
    }

    Write-Step "Checking Python dependencies"
    $importsReady = Test-PythonImports -Python $Python
    if ((-not $ForceInstall) -and $importsReady -and $otherStamps.Count -eq 0) {
        Set-Content -LiteralPath $stampPath -Value (Get-Date).ToString("o") -Encoding UTF8
        Write-Ok "Existing Python dependencies are reusable."
        return
    }

    Write-Step "Installing requirements.txt (first run can take several minutes)"
    Invoke-Python -Python $Python -Arguments @(
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "-r", $RequirementsFile
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependency installation failed. Check the network and retry."
    }

    if (-not (Test-PythonImports -Python $Python)) {
        throw "Python dependencies were installed, but required modules still cannot be imported."
    }

    Get-ChildItem -LiteralPath $RuntimeDir -Filter "python-deps-*.ok" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Set-Content -LiteralPath $stampPath -Value (Get-Date).ToString("o") -Encoding UTF8
    Write-Ok "Python dependencies installed."
}

function Get-NpmCommand {
    foreach ($name in @("npm.cmd", "npm.exe", "npm")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            return $command.Source
        }
    }
    return $null
}

function Test-NodeModules {
    $vite = Join-Path $WebRoot "node_modules\.bin\vite.cmd"
    if (-not (Test-Path -LiteralPath $vite -PathType Leaf) -or -not $Npm) {
        return $false
    }

    Push-Location $WebRoot
    try {
        & $Npm ls --depth=0 *> $null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        Pop-Location
    }
}

function Ensure-Frontend {
    param([bool]$ForceRebuild)

    $script:Npm = Get-NpmCommand
    $distIndex = Join-Path $WebDist "index.html"
    $packageLock = Join-Path $WebRoot "package-lock.json"
    $npmContext = (Get-Content -LiteralPath $packageLock -Raw)
    $npmStamp = Join-Path $RuntimeDir ("npm-deps-" + (Get-TextSha256 -Text $npmContext) + ".ok")

    if ((-not $ForceInstall) -and (Test-Path -LiteralPath $npmStamp -PathType Leaf) -and (Test-NodeModules)) {
        Write-Notice "Frontend dependencies are ready."
    }
    elseif (Test-NodeModules) {
        Set-Content -LiteralPath $npmStamp -Value (Get-Date).ToString("o") -Encoding UTF8
        Write-Ok "Existing frontend dependencies are reusable."
    }
    else {
        if (-not $script:Npm) {
            if (Test-Path -LiteralPath $distIndex -PathType Leaf) {
                Write-Warning "Node.js/npm is unavailable. Using the existing prebuilt frontend."
            }
            else {
                throw "Node.js/npm is required because no frontend build is available. Install Node.js 20+ and retry."
            }
        }
        else {
            Write-Step "Installing frontend dependencies"
            Push-Location $WebRoot
            try {
                & $script:Npm ci
                if ($LASTEXITCODE -ne 0) {
                    throw "npm ci failed."
                }
            }
            finally {
                Pop-Location
            }
            Set-Content -LiteralPath $npmStamp -Value (Get-Date).ToString("o") -Encoding UTF8
            Write-Ok "Frontend dependencies installed."
        }
    }

    $buildInputs = @(
        (Join-Path $WebRoot "index.html"),
        (Join-Path $WebRoot "package.json"),
        $packageLock
    )
    $buildInputs += @(Get-ChildItem -LiteralPath (Join-Path $WebRoot "src") -Recurse -File)
    $newestInput = $buildInputs |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    $needsBuild = $ForceRebuild -or
        (-not (Test-Path -LiteralPath $distIndex -PathType Leaf)) -or
        ($newestInput.LastWriteTime -gt (Get-Item -LiteralPath $distIndex).LastWriteTime.AddSeconds(1))

    if (-not $needsBuild) {
        Write-Notice "Frontend build is up to date."
        return
    }

    if (-not $Npm) {
        throw "Node.js/npm is required to build the frontend. Install Node.js 20+ and retry."
    }

    Write-Step "Building frontend"
    Push-Location $WebRoot
    try {
        & $Npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed."
        }
    }
    finally {
        Pop-Location
    }
    Write-Ok "Frontend build completed."
}

function Get-DemoHealth {
    param([int]$HealthPort)

    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$HealthPort/api/health" -TimeoutSec 2
        if ($health.ok -eq $true -and $health.service -eq "icma-demo") {
            return $health
        }
    }
    catch {
        return $null
    }
    return $null
}

function Test-LoopbackPort {
    param([int]$CandidatePort)

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $CandidatePort)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Open-DemoPage {
    param([string]$Url)

    Write-Ok "ICMA Demo is ready: $Url"
    if ($NoBrowser) {
        Write-Notice "Browser launch skipped because -NoBrowser was supplied."
        return
    }

    try {
        Start-Process -FilePath $Url
        Write-Notice "Opened the demo in the default browser."
    }
    catch {
        Write-Warning "The server is running, but the browser could not be opened automatically. Open $Url manually."
    }
}

try {
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

    $requestedPort = $Port
    if ($requestedPort -eq 0) {
        if ($env:ICMA_DEMO_PORT) {
            $requestedPort = [int]$env:ICMA_DEMO_PORT
        }
        else {
            $requestedPort = 8765
        }
    }
    if ($requestedPort -lt 1 -or $requestedPort -gt 65535) {
        throw "Port must be between 1 and 65535."
    }

    if (Test-Path -LiteralPath $StateFile -PathType Leaf) {
        try {
            $knownState = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
            if ($knownState.port -and $knownState.pid -and (Get-Process -Id ([int]$knownState.pid) -ErrorAction SilentlyContinue)) {
                $health = Get-DemoHealth -HealthPort ([int]$knownState.port)
                if ($health) {
                    Open-DemoPage -Url ([string]$knownState.url)
                    exit 0
                }
            }
        }
        catch {
            Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
        }
    }

    $health = Get-DemoHealth -HealthPort $requestedPort
    if ($health) {
        $url = "http://127.0.0.1:$requestedPort/"
        Open-DemoPage -Url $url
        exit 0
    }

    $selectedPort = $null
    for ($candidatePort = $requestedPort; $candidatePort -le [Math]::Min(65535, $requestedPort + 20); $candidatePort++) {
        if (Test-LoopbackPort -CandidatePort $candidatePort) {
            $selectedPort = $candidatePort
            break
        }
    }
    if (-not $selectedPort) {
        throw "No free localhost port was found near $requestedPort."
    }
    if ($selectedPort -ne $requestedPort) {
        Write-Warning "Port $requestedPort is busy; using $selectedPort instead."
    }

    $python = Get-PythonBinary -ExplicitPath $PythonPath
    Write-Step "Using $($python.Label): Python $($python.Version)"
    Ensure-PythonDependencies -Python $python
    Ensure-Frontend -ForceRebuild ([bool]$Rebuild)

    $stdoutLog = Join-Path $LogDir "demo-server.out.log"
    $stderrLog = Join-Path $LogDir "demo-server.err.log"
    Set-Content -LiteralPath $stdoutLog -Value "" -Encoding UTF8
    Set-Content -LiteralPath $stderrLog -Value "" -Encoding UTF8

    Write-Step "Starting ICMA Demo on port $selectedPort"
    $uvicornArgs = @(
        "-m", "uvicorn", "demo_server.app:app",
        "--host", "127.0.0.1",
        "--port", [string]$selectedPort,
        "--log-level", "info"
    )
    $allArgs = @($python.Prefix) + $uvicornArgs
    $ServerProcess = Start-Process `
        -FilePath $python.FilePath `
        -ArgumentList $allArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru

    $url = "http://127.0.0.1:$selectedPort/"
    $ready = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        Start-Sleep -Milliseconds 500
        $ServerProcess.Refresh()
        if ($ServerProcess.HasExited) {
            break
        }
        if (Get-DemoHealth -HealthPort $selectedPort) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        $errorTail = @()
        if (Test-Path -LiteralPath $stderrLog) {
            $errorTail = @(Get-Content -LiteralPath $stderrLog -Tail 30)
        }
        $message = "The API server did not become ready. Logs: $stderrLog"
        if ($errorTail.Count -gt 0) {
            $message += "`n" + ($errorTail -join "`n")
        }
        if ($ServerProcess -and -not $ServerProcess.HasExited) {
            $taskkill = Get-Command taskkill.exe -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($taskkill) {
                & $taskkill.Source /PID $ServerProcess.Id /T /F | Out-Null
            }
        }
        throw $message
    }

    $state = [PSCustomObject]@{
        pid = $ServerProcess.Id
        port = $selectedPort
        url = $url
        python = $python.FilePath
        startedAt = (Get-Date).ToString("o")
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $StateFile -Encoding UTF8

    Write-Notice "Server PID: $($ServerProcess.Id)"
    Write-Notice "Server log: $stdoutLog"
    Open-DemoPage -Url $url
    exit 0
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
