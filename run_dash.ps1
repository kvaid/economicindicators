Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot
python -m pip install -r requirements.txt
python keyeconomicindicators.py
