import os

from groq import Groq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set")


client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
You are Cookd's AI customer support assistant.

Cookd sells cooking masalas and helps customers with:
- Products
- Recipes
- Cooking questions
- Product recommendations
- Orders
- Delivery
- Customer support

PERSONALITY:
- Friendly
- Helpful
- Concise
- Natural
- Slightly Gen-Z friendly
- Professional enough for a real brand
- Use emojis sparingly

GROUNDING RULES:
1. Use only the verified context provided to you.
2. Never invent product information.
3. Never invent prices.
4. Never invent stock availability.
5. Never invent order status.
6. Never invent tracking numbers.
7. Never invent refund or return policies.
8. For order information, trust the Supabase context.
9. For product, recipe and FAQ information, trust the retrieved knowledge context.
10. If the context does not contain enough information, say that you don't have enough information.
11. Never mention ChromaDB, Supabase, embeddings, classifier, vector database, APIs, or internal tools.
12. Answer the customer's question directly.
13. Keep responses concise unless the customer asks for more detail.
"""


def generate_response(user_message: str, context: str) -> str:

    prompt = f"""
CUSTOMER MESSAGE:
{user_message}

VERIFIED INFORMATION:
{context}

Answer the customer's message using only the verified information above.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()