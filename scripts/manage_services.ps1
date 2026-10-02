param([ValidateSet('start','stop')] [string]$Action)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
# Port 8001 (IndicMT/NLLB) removed — translation now done via Groq LLM API
$ports = @(8000,8002,8003,8004,8005,5173)
$runtime = Join-Path $root 'data\runtime'
$logDir = Join-Path $root 'data\logs'
$pidFile = Join-Path $runtime 'services.json'

function Get-ListenerPids {
  param([int]$Port)
  $lines = netstat -ano -p tcp | Select-String (":$Port\s+.*LISTENING\s+(\d+)\s*$")
  foreach ($line in $lines) { if ($line.Line -match '\s(\d+)\s*$') { [int]$Matches[1] } }
}

function Stop-ProjectListeners {
  foreach ($port in $ports) {
    foreach ($processId in (Get-ListenerPids -Port $port | Sort-Object -Unique)) {
      $process = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
      if (-not $process) { continue }
      $command = [string]$process.CommandLine
      $valid = switch ($port) {
        8000 { $process.Name -eq 'python.exe' -and $command -match 'uvicorn app\.main:app .*--app-dir backend .*--port 8000' }
        8002 { $process.Name -eq 'python.exe' -and $command -match 'uvicorn app\.main:app .*--app-dir tts_service .*--port 8002' }
        8003 { $process.Name -eq 'python.exe' -and $command -match 'uvicorn reranker_service\.app\.main:app .*--port 8003' }
        8004 { $process.Name -eq 'python.exe' -and $command -match 'vector_store[\\/]server\.py' }
        8005 { $process.Name -eq 'python.exe' -and $command -match 'uvicorn app\.main:app .*--app-dir embedding_service .*--port 8005' }
        5173 { $process.Name -eq 'node.exe' -and $command -like "*$root*frontend*node_modules*vite.js*--port 5173*" }
      }
      if (-not $valid) { throw "Port $port is occupied by a process that does not match the project's expected command. PID $processId was left running." }
      Write-Host "Stopping verified project service on port $port (PID $processId)"
      Stop-Process -Id $processId -Force
    }
  }
  if (Test-Path $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
  Start-Sleep -Milliseconds 500
  foreach ($port in $ports) { if (Get-ListenerPids -Port $port) { throw "Port $port is still in use after the controlled shutdown." } }
}

if ($Action -eq 'stop') {
  Stop-ProjectListeners
  Write-Host 'TeluguBridge project services stopped.'
  exit 0
}

Stop-ProjectListeners
New-Item -ItemType Directory -Force -Path $runtime,$logDir | Out-Null
Remove-Item Env:HF_HOME,Env:HF_HUB_CACHE,Env:TRANSFORMERS_CACHE,Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'

$services = @(
  @{name='bge'; port=8005; exe='embedding_service\.venv\Scripts\python.exe'; args=@('-m','uvicorn','app.main:app','--app-dir','embedding_service','--host','127.0.0.1','--port','8005'); cwd=$root},
  @{name='reranker'; port=8003; exe='embedding_service\.venv\Scripts\python.exe'; args=@('-m','uvicorn','reranker_service.app.main:app','--host','127.0.0.1','--port','8003'); cwd=$root},
  @{name='retrieval'; port=8004; exe='vector_store\.venv\Scripts\python.exe'; args=@('vector_store\server.py'); cwd=$root},
  @{name='tts'; port=8002; exe='tts_service\.venv\Scripts\python.exe'; args=@('-m','uvicorn','app.main:app','--app-dir','tts_service','--host','127.0.0.1','--port','8002'); cwd=$root},
  @{name='backend'; port=8000; exe='backend\.venv\Scripts\python.exe'; args=@('-m','uvicorn','app.main:app','--app-dir','backend','--host','127.0.0.1','--port','8000'); cwd=$root}
)
$started = @()
$failures = @()
foreach ($service in $services) {
  $exe = Join-Path $root $service.exe
  if (-not (Test-Path $exe)) { $failures += "Missing $($service.name) runtime: $($service.exe)"; continue }
  $out = Join-Path $logDir "$($service.name).out.log"
  $err = Join-Path $logDir "$($service.name).err.log"
  try { $proc = Start-Process -FilePath $exe -ArgumentList $service.args -WorkingDirectory $service.cwd -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru }
  catch { $failures += "Could not start $($service.name): $($_.Exception.Message)"; continue }
  $started += [pscustomobject]@{name=$service.name; port=$service.port; pid=$proc.Id}
  $started | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
  Write-Host "Started $($service.name) (PID $($proc.Id)); waiting for readiness on $($service.port)."
  $ready = $false
  $deadline = (Get-Date).AddMinutes(20)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
      $tail = if (Test-Path $err) { (Get-Content $err -Tail 30) -join "`n" } else { 'No service log was created.' }
      $failures += "$($service.name) exited during startup. Log tail:`n$tail"
      break
    }
    try {
      $health = Invoke-RestMethod -Uri "http://127.0.0.1:$($service.port)/health" -TimeoutSec 3
      if ($health.status -eq 'ok' -and ($health.loaded -ne $false)) { $ready = $true; break }
    } catch { }
  }
  if (-not $ready -and -not $proc.HasExited) { $failures += "$($service.name) did not become ready on port $($service.port) within 20 minutes." }
  if ($ready) {
    $listener = @(Get-ListenerPids -Port $service.port | Select-Object -First 1)
    if ($listener.Count -gt 0) {
      $started[-1].pid = [int]$listener[0]
      $started | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
    }
  }
}
$frontend = Join-Path $root 'frontend'
if (Test-Path (Join-Path $frontend 'node_modules\vite\bin\vite.js')) {
  $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
  if (-not $nodeCommand) { $failures += 'Node.js was not found; frontend was not started.' }
  else {
  $node = $nodeCommand.Source
  $vite = Join-Path $frontend 'node_modules\vite\bin\vite.js'
  $proc = Start-Process -FilePath $node -ArgumentList @($vite,'--host','127.0.0.1','--port','5173','--strictPort') -WorkingDirectory $frontend -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'frontend.out.log') -RedirectStandardError (Join-Path $logDir 'frontend.err.log') -PassThru
  $started += [pscustomobject]@{name='frontend'; port=5173; pid=$proc.Id}
  $started | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
  $ready = $false
  for ($i=0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -TimeoutSec 2; $ready = $true; break } catch { }
  }
  if (-not $ready) { $failures += 'Frontend did not become ready on port 5173.' }
  else {
    $listener = @(Get-ListenerPids -Port 5173 | Select-Object -First 1)
    if ($listener.Count -gt 0) { $started[-1].pid = [int]$listener[0] }
  }
  }
} else { $failures += 'Frontend dependencies are not installed. Run npm install in frontend first.' }
$started | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
if ($failures.Count) {
  Write-Host 'Project startup completed with blocked services:'
  $failures | ForEach-Object { Write-Host " - $_" }
  exit 1
}
Write-Host 'All project services and the frontend are ready.'
