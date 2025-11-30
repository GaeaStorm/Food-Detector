import os
import json
import unicodedata
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------- CONFIG ----------
DATA_PATH = "dataset/recipe/recipes_with_nutritional_info.json"   # adjust if needed
CHROMA_DIR = "chroma_recipe_db"
COLLECTION_NAME = "recipes_rag"
EMBEDDING_MODEL = "text-embedding-3-small"  # or text-embedding-3-large

client = OpenAI()


def normalize_text(s: str) -> str:
    """Lowercase + strip + basic unicode normalization."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip().lower()


def ingredient_line(ing_text: str, quantity: str | None, unit: str | None) -> str:
    """Build a human-readable ingredient line like '8 ounce yogurt, greek...'."""
    parts = []
    if quantity:
        parts.append(quantity)
    if unit:
        parts.append(unit)
    if ing_text:
        parts.append(ing_text)
    return " ".join(parts)


def load_recipes(path: str) -> List[Dict[str, Any]]:
    print(f"Loading recipes from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} recipes.")
    return data


def make_doc_from_recipe(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn one JSON recipe entry into:
      - document text (for embeddings + context)
      - metadata (for filters)
    """
    rid = rec.get("id")
    title = rec.get("title", "").strip()

    # Ingredients
    ingredient_texts = [ing.get("text", "") for ing in rec.get("ingredients", [])]
    quantities = [q.get("text", "") for q in rec.get("quantity", [])]
    units = [u.get("text", "") for u in rec.get("unit", [])]

    ingredients_lines = []
    for i, ing_text in enumerate(ingredient_texts):
        qty = quantities[i] if i < len(quantities) else None
        unit = units[i] if i < len(units) else None
        ingredients_lines.append("- " + ingredient_line(ing_text, qty, unit))

    # Instructions
    instruction_texts = [step.get("text", "") for step in rec.get("instructions", [])]
    instructions_block = "\n".join(
        f"{i+1}. {txt}" for i, txt in enumerate(instruction_texts) if txt.strip()
    )

    # Nutrition (per 100g)
    nutr = rec.get("nutr_values_per100g", {}) or {}
    kcal = nutr.get("energy")
    protein = nutr.get("protein")
    fat = nutr.get("fat")
    sugars = nutr.get("sugars")
    salt = nutr.get("salt")

    nutrition_lines = []
    if isinstance(kcal, (int, float)):
        nutrition_lines.append(f"- Energy: {kcal:.1f} kcal per 100g")
    if isinstance(protein, (int, float)):
        nutrition_lines.append(f"- Protein: {protein:.1f} g per 100g")
    if isinstance(fat, (int, float)):
        nutrition_lines.append(f"- Fat: {fat:.1f} g per 100g")
    if isinstance(sugars, (int, float)):
        nutrition_lines.append(f"- Sugars: {sugars:.1f} g per 100g")
    if isinstance(salt, (int, float)):
        nutrition_lines.append(f"- Salt: {salt:.3f} g per 100g")

    nutrition_block = "\n".join(nutrition_lines)

    url = rec.get("url", "") or ""

    # Build main document text
    doc_text = f"""Title: {title}

Ingredients:
{os.linesep.join(ingredients_lines)}

Instructions:
{instructions_block or "N/A"}

Nutrition (per 100g):
{nutrition_block or "N/A"}

Source URL: {url}
"""

    # ---- METADATA (NO None VALUES!) ----
    ingredients_norm = [normalize_text(t) for t in ingredient_texts if t.strip()]
    ingred_set = sorted(set(ingredients_norm))

    # Start with only guaranteed-safe values
    metadata: Dict[str, Any] = {
        "id": rid or "",
        "title": title,
        # store ingredient list as JSON string
        "ingredients_json": json.dumps(ingred_set),
        "url": url,
    }

    # Only add numeric fields if present
    if isinstance(kcal, (int, float)):
        metadata["kcal_per_100g"] = float(kcal)
    if isinstance(protein, (int, float)):
        metadata["protein_per_100g"] = float(protein)
    if isinstance(fat, (int, float)):
        metadata["fat_per_100g"] = float(fat)
    if isinstance(sugars, (int, float)):
        metadata["sugars_per_100g"] = float(sugars)
    if isinstance(salt, (int, float)):
        metadata["salt_per_100g"] = float(salt)

    # Cuisine placeholder as empty string (not None) – you can overwrite later
    metadata["cuisine"] = ""

    return {
        "id": rid,
        "text": doc_text,
        "metadata": metadata,
    }


def build_index():
    # Init Chroma
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    recipes = load_recipes(DATA_PATH)

    # You can sample while prototyping (e.g., recipes[:5000])
    batch_size = 64
    to_add_ids = []
    to_add_docs = []
    to_add_metas = []
    to_add_embs = []

    print("Building index...")
    for i, rec in enumerate(recipes):
        try:
            doc = make_doc_from_recipe(rec)
        except Exception as e:
            print(f"Skipping recipe due to error: {e}")
            continue

        if not doc["id"]:
            continue

        to_add_ids.append(doc["id"])
        to_add_docs.append(doc["text"])
        to_add_metas.append(doc["metadata"])

        # Embed this document
        emb_resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[doc["text"]],
        )
        embedding = emb_resp.data[0].embedding
        to_add_embs.append(embedding)

        # Batch insert for performance
        if len(to_add_ids) >= batch_size:
            collection.add(
                ids=to_add_ids,
                documents=to_add_docs,
                metadatas=to_add_metas,
                embeddings=to_add_embs,
            )
            print(f"Indexed {i+1} recipes...")
            to_add_ids, to_add_docs, to_add_metas, to_add_embs = [], [], [], []

    # Flush remaining
    if to_add_ids:
        collection.add(
            ids=to_add_ids,
            documents=to_add_docs,
            metadatas=to_add_metas,
            embeddings=to_add_embs,
        )
        print(f"Indexed final {len(to_add_ids)} recipes.")

    print("✅ Index build complete.")


if __name__ == "__main__":
    build_index()
