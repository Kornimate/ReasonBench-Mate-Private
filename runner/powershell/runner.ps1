$pythonArgs = @(
    "runner/runner.py"
    "--provider", "openai"
    "--api_key", "OPENAI_API_KEY_CLAN"
    "--model", "gpt-4.1-nano"
    "--model_config_path", "models_config.yaml"
)

python @pythonArgs
