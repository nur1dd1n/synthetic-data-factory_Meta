from __future__ import annotations

import os
import json
from openai import OpenAI


def explain_dataset(profile: dict, trust_report: dict | None = None) -> str:
    api_key = os.getenv("MODEL_API_KEY", "").strip()

    if not api_key or "your_" in api_key or "replace_with" in api_key:
        raise RuntimeError(
            "MODEL_API_KEY не найден. Проверь файл .env"
        )

    client = OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
    )

    report_context = (
        "\n\nИзмеренный агрегированный отчёт после генерации:\n"
        + json.dumps(trust_report, ensure_ascii=False, default=str)
        if trust_report is not None
        else "\n\nСинтетические данные ещё не генерировались."
    )

    response = client.responses.create(
        model="muse-spark-1.3",
        input=(
            "Ты — эксперт по качеству данных и машинному обучению.\n"
            "Проанализируй профиль датасета и:\n"
            "1. Объясни проблемы качества данных.\n"
            "2. Если есть измеренный отчёт, ссылайся только на его реальные метрики.\n"
            "3. Дай не более пяти конкретных следующих шагов.\n"
            "4. Не выдумывай метрики и не утверждай, что данные приватны.\n\n"
            "Профиль датасета:\n"
            + json.dumps(profile, ensure_ascii=False, default=str)
            + report_context
        ),
    )

    return response.output_text
