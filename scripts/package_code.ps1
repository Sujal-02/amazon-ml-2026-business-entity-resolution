# Zip the code folder (no caches, no data) -> ..\..\ber_code.zip  (run from anywhere)
$repo = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $repo
$tmp = Join-Path $env:TEMP "ber_pkg"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force (Join-Path $tmp "business_entity_resolution") | Out-Null
robocopy $repo (Join-Path $tmp "business_entity_resolution") /E /XD __pycache__ .pytest_cache .ipynb_checkpoints /XF *.pyc | Out-Null
# LF line endings for shell scripts (they run on Linux)
Get-ChildItem -Recurse (Join-Path $tmp "business_entity_resolution\scripts") -Filter *.sh | ForEach-Object {
  $t = [IO.File]::ReadAllText($_.FullName) -replace "`r`n", "`n"; [IO.File]::WriteAllText($_.FullName, $t) }
$zip = Join-Path $root "ber_code.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
# bsdtar (built into Windows 10+) writes a standard zip with forward-slash paths (Compress-Archive on PS 5.1 does not)
& tar.exe -a -c -f $zip -C $tmp business_entity_resolution
Write-Host "wrote $zip"
