$benchmark = "mtsamples_procedures"
$method = "reagents"
$split = "single"

$provider = "openai"
$apiKey = "OPENAI_API_KEY"
$model = "gpt-4.1-nano"

$configPath = Join-Path "scripts/configs" "$benchmark.env"
Get-Content -Path $configPath | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
        return
    }

    $key, $value = $_ -split '=', 2
    Set-Variable -Name $key -Value $value
}

if ($method -in @("io", "cot")) {
    $MAX_COMPLETION_TOKENS = 10000
}

$pythonArgs = @(
    "scripts/simple/simple.py"
    "--benchmark", $benchmark
    "--method", $method
    "--model", $model
    "--batch_size", "1"
    "--timeout", "2.0"
    "--temperature", $TEMPERATURE
    "--max_completion_tokens", "$MAX_COMPLETION_TOKENS"
    "--top_p", $TOP_P
    "--dataset_path", "datasets/dataset_${benchmark}.csv.gz"
    "--split", $split
    "--correctness", "1"
    "--allow_batch_overflow", "1"
    "--ns_ratio", "0.0"
    "--provider", $provider
    "--api_key", $apiKey
)

if (-not [string]::IsNullOrWhiteSpace($STOP)) {
    $pythonArgs += @("--stop", $STOP)
}

$pythonArgs += "--value_cache"

python @pythonArgs
