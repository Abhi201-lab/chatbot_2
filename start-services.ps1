# Simple helper to start each service in its own PowerShell window.
# Run: powershell -ExecutionPolicy Bypass -File .\start-services.ps1

$root = "C:\Users\jeeta\Documents\chatbot"

function Start-ServiceWindow($workdir, $cmd) {
    if (-not (Test-Path $workdir)) {
        Write-Host "Directory not found: $workdir" -ForegroundColor Yellow
        return
    }

    $activateCmd = @'
if (Test-Path .\.venv\Scripts\Activate) { . .\.venv\Scripts\Activate } elseif (Test-Path .\venv\Scripts\Activate) { . .\venv\Scripts\Activate }
'@

    $fullCmd = "Set-Location -Path '$workdir'; $activateCmd ; $cmd"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command",$fullCmd
}

# Adjust the commands/ports if your services use different entrypoints or ports.
Start-ServiceWindow "$root\chat_api_v1"  "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
Start-ServiceWindow "$root\kmapi_v1"     "uvicorn main:app --reload --host 0.0.0.0 --port 8001"
Start-ServiceWindow "$root\llmapi_v1"    "uvicorn main:app --reload --host 0.0.0.0 --port 8002"
Start-ServiceWindow "$root\ingestapi_v1" "uvicorn main:app --reload --host 0.0.0.0 --port 8003"
Start-ServiceWindow "$root\chatui_v1"    "streamlit run main.py --server.port 8501"

Write-Host "Start commands issued. Check individual windows for logs." -ForegroundColor Green