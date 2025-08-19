from typing import Any, Dict, List
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')


async def generate_routine_for_user(db, clerk_user_id: str) -> Dict[str, Any]:
    user = await db.users.find_one({"clerk_user_id": clerk_user_id})
    if not user:
        raise ValueError("User not found")
    latest_analysis = await db.skin_analyses.find_one({"clerk_user_id": clerk_user_id}, sort=[["created_at", -1]])
    if not latest_analysis:
        raise ValueError("No skin analysis found for user")

    age = user.get("age")
    gender = user.get("gender")
    sensitive_skin = user.get("sensitive_skin")
    skinType = latest_analysis.get("comprehensive", {}).get("skinType")
    metrics = latest_analysis.get("metrics", {})

    metrics_summary_lines: List[str] = []
    for name, data in metrics.items():
        if isinstance(data, dict):
            sev = data.get("severity")
            score = data.get("score")
            parts: List[str] = []
            if sev:
                parts.append(f"severity={sev}")
            if isinstance(score, (float, int)):
                parts.append(f"score={score:.2f}")
            if parts:
                metrics_summary_lines.append(f"- {name}: " + ", ".join(parts))

    prompt = f"""
    You are a dermatologist assistant. Create a complete skincare routine JSON based on the following profile and skin data.

    Profile:
    - Age: {age}
    - Gender: {gender}
    - Sensitive Skin: {sensitive_skin}
    - Skin Type: {skinType}

    Skin Metrics:
    {chr(10).join(metrics_summary_lines)}

    Return STRICT JSON with this schema (no commentary):
    {{
      "skinType": string,
      "age": number,
      "gender": string,
      "sensitiveSkin": boolean,
      "routine": {{
        "morning": [{{"step": number, "product": string, "description": string, "duration": string, "ingredients": [string]}}],
        "evening": [{{"step": number, "product": string, "description": string, "duration": string, "ingredients": [string]}}],
        "weekly": [{{"frequency": string, "product": string, "description": string, "ingredients": [string]}}]
      }},
      "ingredients": [{{
        "name": string,
        "benefits": string,
        "pros": [string],
        "usage": string,
        "warnings": [string]
      }}]
    }}
    Make the products and ingredients appropriate for the provided age, gender, skin type, sensitivity, and metrics. Prefer gentle options when metrics indicate sensitivity or redness.
    """

    response = model.generate_content(prompt)

    import json
    text = (response.text or "").strip()
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON in model response")
    data = json.loads(text[start:end])

    record = {
        "clerk_user_id": clerk_user_id,
        "created_at": datetime.utcnow(),
        "source_analysis_id": str(latest_analysis.get("_id")),
        "routine": data,
    }
    await db.skin_routines.insert_one(record)
    return data 