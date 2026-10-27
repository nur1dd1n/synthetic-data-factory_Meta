import os
import json
from openai import OpenAI


def explain_dataset(profile: dict) -> str:
    api_key = os.getenv("MODEL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MODEL_API_KEY не найден. Проверь файл .env"
        )

    client = OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
    )

    response = client.responses.create(
        model="muse-spark-1.3",
        input=(
            "Ты — эксперт по качеству данных и машинному обучению.\n"
            "Проанализируй профиль датасета и:\n"
            "1. Объясни проблемы качества данных.\n"
            "2. Предложи метод генерации синтетических данных.\n"
            "3. Посоветуй проверки качества и приватности.\n"
            "4. Не утверждай, что данные приватны без доказательств.\n\n"
            "Профиль датасета:\n"
            + json.dumps(profile, ensure_ascii=False, default=str)
        ),
    )

    return response.output_text