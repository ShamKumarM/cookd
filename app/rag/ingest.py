import json, os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
client=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH",str(ROOT/"chroma_data")))
collection=client.get_or_create_collection(os.getenv("CHROMA_COLLECTION","cookd_knowledge"))
model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))

docs=[]
for p in json.loads((DATA/"products.json").read_text(encoding="utf-8")):
    docs.append({
      "id":"product_"+p["id"],
      "text":f"""Cookd Product
Name: {p["name"]}
Category: {p["category"]}
Price: ₹{p["price"]}
Pack options: {", ".join(p["size_options"])}
Description: {p["description"]}
Serves: {p.get("serves") or "not specified"}
SKU: {p["sku"]}""",
      "metadata":{"type":"product","product_id":p["id"],"sku":p["sku"],"category":p["category"],"source_url":p["url"]}
    })

for r in json.loads((DATA/"recipes.json").read_text(encoding="utf-8")):
    docs.append({
      "id":"recipe_"+r["id"],
      "text":f"""Cookd Recipe
Recipe: {r["title"]}
Cuisine: {r["cuisine"]}
Meal type: {r["meal_type"]}
Protein: {r["protein"]}
Spice: {r["spice"]}
Time: {r["total_min"]} minutes
Servings: {r["servings"]}
Description: {r["description"]}
Cookd products: {", ".join(r["product_match"]) if r["product_match"] else "None specified"}""",
      "metadata":{"type":"recipe","recipe_id":r["id"],"cuisine":r["cuisine"],"protein":r["protein"],"source_url":r["url"]}
    })

for f in json.loads((DATA/"faqs.json").read_text(encoding="utf-8")):
    docs.append({
      "id":f["id"],
      "text":f"Cookd FAQ\nQuestion: {f['question']}\nAnswer: {f['answer']}",
      "metadata":{"type":"faq","category":f["category"]}
    })

emb=model.encode([d["text"] for d in docs],normalize_embeddings=True).tolist()
collection.upsert(ids=[d["id"] for d in docs],documents=[d["text"] for d in docs],
                  metadatas=[d["metadata"] for d in docs],embeddings=emb)
print(f"Indexed {len(docs)} documents.")
