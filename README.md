# Cookd AI RAG + Intent Router MVP

## Architecture

WhatsApp/User -> Intent Classifier -> Router

RAG intents:
- product_question -> Chroma product knowledge
- recipe_finding -> Chroma recipe knowledge
- recommendation -> Chroma mixed retrieval (+ later Supabase stock check)
- general_faq -> Chroma FAQ knowledge

Tool intents:
- order_tracking -> Supabase tools
- feedback_complaint -> Supabase support tools
- return_refund -> Supabase tools
- cart_action -> ecommerce/Supabase tools

## Chroma should contain

- Cookd product descriptions
- product categories and variants
- recipe descriptions
- cuisine/protein/spice/time/servings
- Cookd FAQs and cooking guidance
- later: approved brand/product education

## Chroma should NOT contain

- customer phone/address
- order status
- payment status
- live inventory
- refunds
- carts
- support ticket state

Those remain in Supabase and are accessed through controlled tools.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app/classifier/train.py
python app/rag/ingest.py

uvicorn app.main:app --reload
```

Test:
- GET /health
- POST /search {"message":"Which product is good for fish fry?"}
- POST /route {"message":"Where is my order?"}

Important: the classifier only routes. It must never directly execute a database action. The agent/tool layer makes that decision and validates permissions.
