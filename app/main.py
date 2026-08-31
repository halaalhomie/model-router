from google import genai

from app.config import load_settings


PROMPT = "Explain, in one sentence, what an LLM model router does."


def main() -> None:
    settings = load_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=PROMPT,
    )

    print(response.text)


if __name__ == "__main__":
    main()
