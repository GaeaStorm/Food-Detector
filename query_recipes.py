import os
import json
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

CHROMA_DIR = "chroma_recipe_db"
COLLECTION_NAME = "recipes_rag"
EMBEDDING_MODEL = "text-embedding-3-large"  # must match build_recipe_index

client = OpenAI()


def normalize(s: str) -> str:
    return s.strip().lower()


def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(COLLECTION_NAME)


def retrieve_candidates(
    ingredients: List[str],
    k: int = 50,
) -> Dict[str, Any]:
    """
    Use semantic search to pull candidate recipes.
    We'll apply stricter ingredient/nutrition filters after retrieval.
    """
    collection = load_collection()

    query_text = (
        "Find recipes that use one or more of these ingredients: "
        + ", ".join(ingredients)
        + ". Prefer recipes that match as many of these ingredients as possible."
    )

    emb_resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text],
    )
    query_emb = emb_resp.data[0].embedding

    res = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return res


def filter_candidates(
    res: Dict[str, Any],
    ingredients: List[str],
    max_kcal_per_100g: Optional[float] = None,
    cuisine: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Apply simple Python-side filters based on:
      - loose ingredient overlap (substring match)
      - optional kcal upper bound
      - optional cuisine match (if metadata populated)
    """
    # normalized query terms
    ing_norm = {normalize(x) for x in ingredients if x.strip()}
    out: List[Dict[str, Any]] = []

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    def has_ingredient_match(meta_ings: List[str], query_terms: set[str]) -> bool:
        """Return True if any query term appears (as substring) in any ingredient."""
        if not query_terms:
            return True
        for ing in meta_ings:
            for q in query_terms:
                # both are already lowercased
                if q in ing or ing in q:
                    return True
        return False

    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        # ingredients are stored as a JSON-encoded list of normalized strings
        ing_json = meta.get("ingredients_json", "[]")
        try:
            meta_ingredients = json.loads(ing_json)
        except Exception:
            meta_ingredients = []

        # Require at least 1 overlapping ingredient (substring-based)
        if not has_ingredient_match(meta_ingredients, ing_norm):
            continue

        # Nutrition filter
        if max_kcal_per_100g is not None:
            kcal = meta.get("kcal_per_100g")
            if kcal is None or kcal > max_kcal_per_100g:
                continue

        # Cuisine filter (only if you later populate this field)
        if cuisine:
            m_cuisine = meta.get("cuisine", "")
            if cuisine.lower() not in str(m_cuisine).lower():
                continue

        out.append(
            {
                "id": rid,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )

    # sort by distance (smaller is better)
    out.sort(key=lambda x: x["distance"])
    return out


def suggest_recipes_raw(
    ingredients: List[str],
    max_kcal_per_100g: Optional[float] = None,
    cuisine: Optional[str] = None,
    n_results: int = 50,
    max_return: int = 10,
) -> List[Dict[str, Any]]:
    """
    Main entry point (no LLM):
      1) retrieve candidates with vector search
      2) filter by ingredients / nutrition / cuisine
      3) return top matches as raw recipe docs + metadata

    Returns: list of dicts like:
      {
        "id": "<recipe_id>",
        "title": "...",
        "url": "...",
        "kcal_per_100g": float | None,
        "ingredients": [str, ...],
        "raw_text": "<full doc text for display>"
      }
    """
    res = retrieve_candidates(ingredients, k=n_results)
    filtered = filter_candidates(
        res,
        ingredients=ingredients,
        max_kcal_per_100g=max_kcal_per_100g,
        cuisine=cuisine,
    )

    results: List[Dict[str, Any]] = []
    for item in filtered[:max_return]:
        meta = item["metadata"]
        ing_json = meta.get("ingredients_json", "[]")
        try:
            meta_ingredients = json.loads(ing_json)
        except Exception:
            meta_ingredients = []

        results.append(
            {
                "id": meta.get("id"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "kcal_per_100g": meta.get("kcal_per_100g"),
                "ingredients": meta_ingredients,
                "raw_text": item["text"],
                "distance": item["distance"],
            }
        )

    return results


def _print_results(results: List[Dict[str, Any]]):
    if not results:
        print("No matching recipes found.")
        return

    for i, r in enumerate(results, start=1):
        print("=" * 60)
        print(f"{i}. {r['title']}  (id: {r['id']})")
        if r["kcal_per_100g"] is not None:
            print(f"   ~{r['kcal_per_100g']:.1f} kcal per 100g")
        if r["url"]:
            print(f"   URL: {r['url']}")
        print("   Ingredients (normalized):")
        for ing in r["ingredients"]:
            print(f"     - {ing}")
        # Uncomment to see full text (ingredients+instructions+nutrition)
        # print("\nFull recipe doc:\n")
        # print(r["raw_text"])


if __name__ == "__main__":
    # Example usage:
    example_ingredients = ['brinjal', 'eggplant', 'aubergine', 'capsicum', 'bell pepper', 'green pepper', 'red pepper', 'coriander', 'cilantro', 'garlic', 'onion', 'yellow onion', 'red onion', 'potato', 'potatoes', 'tomato', 'tomatoes']

    results = suggest_recipes_raw(
        ingredients=example_ingredients,
        max_kcal_per_100g=None,  # e.g. 150 to filter by calories
        cuisine=None,            # e.g. "italian" if you add cuisine labels later
        n_results=50,
        max_return=5,
    )
    _print_results(results)