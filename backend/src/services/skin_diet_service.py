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


async def generate_diet_for_user(db, clerk_user_id: str) -> Dict[str, Any]:
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
    concerns = latest_analysis.get("comprehensive", {}).get("concerns", [])
    metrics = latest_analysis.get("metrics", {})

    metrics_summary_lines: List[str] = []
    for name, data in metrics.items():
        if isinstance(data, dict):
            sev = data.get("severity")
            parts: List[str] = []
            if sev:
                parts.append(f"severity={sev}")
            if parts:
                metrics_summary_lines.append(f"- {name}: " + ", ".join(parts))

    prompt = f"""
    You are a dermatologist-nutrition assistant specializing in Indian cuisine. Create a skin-friendly nutrition plan JSON for this profile:
    Age: {age}, Gender: {gender}, Sensitive Skin: {sensitive_skin}, Skin Type: {skinType}
    Key concerns: {', '.join(concerns)}
    Skin metrics summary:\n{chr(10).join(metrics_summary_lines)}

    Return STRICT JSON (no commentary) with Indian food focus:
    {{
      "goals": [string],
      "foodsToEat": {{
        "grains": [{{"food": string, "benefits": string, "nutrients": [string]}}],
        "vegetables": [{{"food": string, "benefits": string, "nutrients": [string]}}],
        "fruits": [{{"food": string, "benefits": string, "nutrients": [string]}}],
        "proteins": [{{"food": string, "benefits": string, "nutrients": [string]}}],
        "dals_legumes": [{{"food": string, "benefits": string, "nutrients": [string]}}],
        "spices_herbs": [{{"food": string, "benefits": string, "nutrients": [string]}}],
        "dairy_alternatives": [{{"food": string, "benefits": string, "nutrients": [string]}}]
      }},
      "supplements": [{{"name": string, "dosage": string, "benefits": string}}],
      "hydration": {{"waterGoal": string, "tips": [string]}},
      "foodsToAvoid": [string],
      "skinBeneficialFoods": [string],
      "cookingTips": [string]
    }}
    
    Focus on:
    - Traditional Indian foods (dal, sabzi, roti, rice, etc.)
    - Ayurvedic principles for skin health
    - Anti-inflammatory Indian spices (turmeric, neem, etc.)
    - Seasonal Indian fruits and vegetables
    - Gentle foods for sensitive skin if applicable
    - Avoid common Indian triggers for acne/inflammation (excess oil, refined sugars, etc.)
    """

    response = model.generate_content(prompt)

    import json
    text = (response.text or "").strip()
    start = text.find('{'); end = text.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON in model response")
    data = json.loads(text[start:end])

    record = {
        "clerk_user_id": clerk_user_id,
        "created_at": datetime.utcnow(),
        "source_analysis_id": str(latest_analysis.get("_id")),
        "diet": data,
    }
    await db.skin_diets.insert_one(record)
    return data 