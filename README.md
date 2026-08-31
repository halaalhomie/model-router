# Intelligent LLM Model Router & Evaluation Platform

## Phase 1: one LLM request

This first milestone verifies the basic synchronous flow:

`Python script -> Gemini API -> model response`

1. Copy `.env.example` to `.env`.
2. Replace `replace_with_your_api_key` with your Gemini API key.
3. Install dependencies: `venv\\Scripts\\python.exe -m pip install -r requirements.txt`
4. Run: `venv\\Scripts\\python.exe -m app.main`

Keep `.env` private. It is intentionally ignored by Git.
