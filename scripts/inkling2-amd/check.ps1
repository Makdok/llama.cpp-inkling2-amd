# One-shot check of the Inkling 2 changes of this branch, with llama-bench on the GPU:
#   1) self-check of every fused pattern against the generic ops on the real model data (prompt of 256 + 24 generated
#      tokens, then generation in small batches like the verify batches of a drafter);
#   2) what one generated token launches (trace);
#   3) speed with and without the fused kernels of this branch (llama-bench, pp2048 and tg128, -ub 512).
# Every run has a time guard: a kernel variant that is not compiled for the device prints "no device code" and then hangs
# at 0 % GPU. Written for Windows PowerShell 5.1 and PowerShell 7 (Linux use is untested).
# Usage: .\check.ps1 -Model <Inkling 2 Mini .gguf> [-Bin <folder of llama-bench>] [-OutDir <folder>] [-NoSpeed]
#                    [-Mask <bits>] [-Env @{INKLING2_N_RS_SEQ='3'}] [-VerifyBatches 3,4,8] [-TimeoutSec 300]
#   -Model may also come from the environment variable INKLING_MODEL, -Bin from INKLING_BIN
#   -Env @{INKLING2_N_RS_SEQ='3'}  builds the graph that llama-server uses for the target model with a drafter of n-max 3
#   -VerifyBatches                 the self-check also runs on generation in batches of that many tokens (@() to skip)
#   -Mask                          GGML_CUDA_INK2_FUSE_MASK for the whole check (default: every fused kind)
param([string]$Model = $env:INKLING_MODEL,
      [string]$Bin = $(if ($env:INKLING_BIN) { $env:INKLING_BIN } else { Join-Path $PSScriptRoot '..\..\build-hip\bin' }),
      [string]$OutDir = (Join-Path ([IO.Path]::GetTempPath()) 'inkling2-check'), [switch]$NoSpeed, [string]$Mask = '',
      [hashtable]$Env = @{}, [int[]]$VerifyBatches = @(4), [int]$TimeoutSec = 300)
if (-not $Model) { throw 'give -Model <path of the .gguf> or set INKLING_MODEL' }
$onWindows = $env:OS -eq 'Windows_NT'
$exe = Join-Path $Bin $(if ($onWindows) { 'llama-bench.exe' } else { 'llama-bench' })
foreach ($f in $exe, $Model) { if (-not (Test-Path $f)) { throw "not found: $f" } }
New-Item -ItemType Directory -Force $OutDir | Out-Null
if ($onWindows -and $env:HIP_PATH) { $env:PATH = (Join-Path $env:HIP_PATH 'bin') + ';' + $env:PATH }
foreach ($k in $Env.Keys) { Set-Item "env:$k" $Env[$k] }

function Invoke-Guarded([string[]]$a, [hashtable]$e, [string]$out) {
    foreach ($k in $e.Keys) { Set-Item "env:$k" $e[$k] }
    $sp = @{ FilePath = $exe; ArgumentList = $a; RedirectStandardOutput = $out; RedirectStandardError = "$out.err"; PassThru = $true }
    if ($onWindows) { $sp.WindowStyle = 'Hidden' }
    $p = Start-Process @sp
    $null = $p.Handle   # Windows PowerShell 5.1 loses the exit code unless the handle is opened while the process runs
    $ok = $p.WaitForExit($TimeoutSec * 1000)
    if (-not $ok) { $p | Stop-Process -Force }
    else { $p.WaitForExit() }
    foreach ($k in $e.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
    if ($ok) { "exit $($p.ExitCode)" } else { "HUNG, killed after $TimeoutSec s" }
}
function Show-Lines([string[]]$files, [string]$pattern) {
    Get-Content $files -ErrorAction SilentlyContinue | Select-String $pattern | ForEach-Object {
        $l = $_.Line -replace 'ink2 verify ', ''
        '  ' + $l.Substring(0, [math]::Min(150, $l.Length))
    }
}

# 1) self-check
$e = @{ GGML_CUDA_INK2_VERIFY = '1'; GGML_CUDA_DISABLE_GRAPHS = '1' }
if ($Mask) { $e.GGML_CUDA_INK2_FUSE_MASK = $Mask }
$log = Join-Path $OutDir 'verify.log'
$r = Invoke-Guarded @('-m', $Model, '-ngl', '99', '-fa', '1', '-p', '256', '-n', '24', '-r', '1') $e $log
"self-check: $r"
Show-Lines @("$log.err", $log) 'ink2 verify|ASSERT|error'
foreach ($vb in $VerifyBatches) {
    $e2 = $e.Clone(); $e2.LLAMA_BENCH_GEN_BATCH = "$vb"
    $log = Join-Path $OutDir "verify_b$vb.log"
    $r = Invoke-Guarded @('-m', $Model, '-ngl', '99', '-fa', '1', '-p', '0', '-n', "$(4 * $vb)", '-r', '1', '-d', '64') $e2 $log
    "self-check, batches of ${vb}: $r"
    Show-Lines @("$log.err") 'ink2 verify|ASSERT|error'
}

# 2) launches of one generated token
$e = @{ GGML_CUDA_INK2_TRACE = '1'; GGML_CUDA_DISABLE_GRAPHS = '1' }
if ($Mask) { $e.GGML_CUDA_INK2_FUSE_MASK = $Mask }
$log = Join-Path $OutDir 'trace.log'
$r = Invoke-Guarded @('-m', $Model, '-ngl', '99', '-fa', '1', '-p', '0', '-n', '8', '-r', '1') $e $log
$t = @(Get-Content "$log.err" -ErrorAction SilentlyContinue | Select-String 'ink2 trace' | ForEach-Object { $_.Line -replace 'ink2 trace: ', '' })
"trace: $r ; " + ($t | Select-Object -First 1)
'  ' + (($t | Select-Object -Skip 1 -First 16 | ForEach-Object { $_.Trim() -replace '\s+', ' ' }) -join ' | ')

# 3) speed with and without the fused kernels of this branch
if (-not $NoSpeed) {
    $with = if ($Mask) { @{ GGML_CUDA_INK2_FUSE_MASK = $Mask } } else { @{} }
    foreach ($v in @(@{ n = 'without the fusions of this branch'; e = @{ GGML_CUDA_INK2_NO_FUSE = '1' } }, @{ n = 'with'; e = $with })) {
        $log = Join-Path $OutDir 'speed.log'
        $r = Invoke-Guarded @('-m', $Model, '-ngl', '99', '-fa', '1', '-p', '2048', '-n', '128', '-b', '2048', '-ub', '512', '-r', '2', '-o', 'jsonl') $v.e $log
        $j = @(Get-Content $log -ErrorAction SilentlyContinue | Where-Object { $_ -match '"avg_ts"' } | ForEach-Object { $_ | ConvertFrom-Json })
        "speed $($v.n): $r ; " + (($j | ForEach-Object { $(if ($_.n_gen -gt 0) { 'tg128' } else { 'pp2048' }) + ' ' + [math]::Round($_.avg_ts, 1) + ' tok/s' }) -join ' ; ')
    }
}
foreach ($k in $Env.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
"logs in $OutDir"
