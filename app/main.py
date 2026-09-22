from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES FIRST
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from app.llm.groq_client import generate_response
from app.classifier.predict import IntentClassifier
from app.rag.retriever import CookdRetriever
from app.router import route

from app.tools.supabase_tools import (
    get_customer_by_phone,
    get_customer_orders,
    get_latest_order,
    get_order_status,
    get_product_stock,
    search_products,
    debug_customers,
    track_latest_order_by_phone,
)


# ---------------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------------

app = FastAPI(title="Cookd AI RAG MVP")

classifier = IntentClassifier()
retriever = CookdRetriever()


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    customer_phone: str | None = None


# ---------------------------------------------------------
# RAG CONTEXT BUILDER
# ---------------------------------------------------------

def build_rag_context(results):

    if not results:
        return "No relevant information was found."

    context_parts = []

    for i, result in enumerate(results, start=1):

        text = result.get("text", "")
        metadata = result.get("metadata", {})

        context_parts.append(
            f"""
SOURCE {i}

CONTENT:
{text}

METADATA:
{metadata}
"""
        )

    return "\n".join(context_parts)


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# MAIN CHAT ENDPOINT
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    # -----------------------------------------------------
    # 1. CLASSIFY USER MESSAGE
    # -----------------------------------------------------

    prediction = classifier.predict(request.message)

    intent = prediction["intent"]
    confidence = prediction["confidence"]


    # =====================================================
    # 2. ORDER TRACKING → SUPABASE → GROQ
    # =====================================================

    if intent == "order_tracking":

        # Customer phone required
        if not request.customer_phone:

            return {
                "success": False,
                "intent": intent,
                "confidence": confidence,
                "error": "customer_phone_required",
                "message": (
                    "Please provide your phone number "
                    "so I can check your order."
                )
            }

        # Get customer + latest order
        result = track_latest_order_by_phone(
            request.customer_phone
        )

        # Customer/order not found
        if not result["success"]:

            return {
                "success": False,
                "intent": intent,
                "confidence": confidence,
                "error": result["error"]
            }

        customer = result["customer"]
        order = result["order"]

        # Build verified Supabase context
        context = f"""
CUSTOMER INFORMATION:

Name: {customer.get("name")}
City: {customer.get("city")}


ORDER INFORMATION:

Order ID: {order.get("id")}
Status: {order.get("status")}
Total: ₹{order.get("total_inr")}
Tracking Number: {order.get("tracking_number")}
Order Created: {order.get("created_at")}
Last Updated: {order.get("updated_at")}
"""

        # Send verified information to Groq
        answer = generate_response(
            user_message=request.message,
            context=context
        )

        return {
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "source": "supabase",
            "answer": answer,
            "order": order
        }


    # =====================================================
    # 3. PRODUCT QUESTION → CHROMA → GROQ
    # =====================================================

    if intent == "product_question":

        results = retriever.search(
            request.message,
            top_k=5,
            doc_type="product"
        )

        context = build_rag_context(results)

        answer = generate_response(
            user_message=request.message,
            context=context
        )

        return {
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "source": "chroma",
            "answer": answer,
            "retrieved_results": results
        }


    # =====================================================
    # 4. RECIPE FINDING → CHROMA → GROQ
    # =====================================================

    if intent == "recipe_finding":

        results = retriever.search(
            request.message,
            top_k=5,
            doc_type="recipe"
        )

        context = build_rag_context(results)

        answer = generate_response(
            user_message=request.message,
            context=context
        )

        return {
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "source": "chroma",
            "answer": answer,
            "retrieved_results": results
        }


    # =====================================================
    # 5. GENERAL FAQ → CHROMA → GROQ
    # =====================================================

    if intent == "general_faq":

        results = retriever.search(
            request.message,
            top_k=5,
            doc_type="faq"
        )

        context = build_rag_context(results)

        answer = generate_response(
            user_message=request.message,
            context=context
        )

        return {
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "source": "chroma",
            "answer": answer,
            "retrieved_results": results
        }


    # =====================================================
    # 6. RECOMMENDATION → CHROMA → GROQ
    # =====================================================

    if intent == "recommendation":

        results = retriever.search(
            request.message,
            top_k=5
        )

        context = build_rag_context(results)

        answer = generate_response(
            user_message=request.message,
            context=context
        )

        return {
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "source": "chroma",
            "answer": answer,
            "retrieved_results": results
        }


    # =====================================================
    # 7. FALLBACK
    # =====================================================

    return {
        "success": True,
        "intent": intent,
        "confidence": confidence,
        "source": "unknown",
        "message": (
            "I need a little more information "
            "to help with that."
        )
    }


# =========================================================
# TEST: GROQ
# =========================================================

@app.get("/test/groq")
def test_groq():

    answer = generate_response(
        user_message="What is Cookd?",
        context="""
Cookd is a cooking masala brand.
It provides cooking products and recipes.
"""
    )

    return {
        "success": True,
        "answer": answer
    }


# =========================================================
# TEST: CUSTOMERS
# =========================================================

@app.get("/test/customers")
def test_customers():

    customers = debug_customers()

    return {
        "customers": customers
    }


# =========================================================
# TEST: ORDER TRACKING
# =========================================================

@app.post("/test/order-tracking")
def test_order_tracking(request: ChatRequest):

    result = track_latest_order_by_phone(
        request.customer_phone
    )

    return result


# =========================================================
# TEST: CUSTOMER BY PHONE
# =========================================================

@app.get("/test/customer/{phone}")
def test_customer(phone: str):

    print("PHONE RECEIVED:", repr(phone))

    customer = get_customer_by_phone(phone)

    return {
        "received_phone": repr(phone),
        "customer": customer
    }


# =========================================================
# TEST: CUSTOMER ORDERS
# =========================================================

@app.get("/test/orders/{customer_id}")
def test_orders(customer_id: str):

    orders = get_customer_orders(customer_id)

    return {
        "orders": orders
    }


# =========================================================
# TEST: LATEST ORDER
# =========================================================

@app.get("/test/latest-order/{customer_id}")
def test_latest_order(customer_id: str):

    order = get_latest_order(customer_id)

    return {
        "order": order
    }


# =========================================================
# ROUTER TEST
# =========================================================

@app.post("/route")
def route_message(req: ChatRequest):

    pred = classifier.predict(req.message)

    decision = route(
        pred["intent"],
        pred["confidence"]
    )

    out = {
        "message": req.message,
        "classification": pred,
        "routing": decision
    }

    if decision["knowledge_source"]:

        out["retrieval"] = retriever.search(
            req.message,
            5,
            None
            if decision["knowledge_source"] == "mixed"
            else decision["knowledge_source"]
        )

    return out


# =========================================================
# SEARCH TEST
# =========================================================

@app.post("/search")
def search(req: ChatRequest):

    return {
        "query": req.message,
        "results": retriever.search(
            req.message,
            5
        )
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )