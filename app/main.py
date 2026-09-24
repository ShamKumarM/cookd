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
from app.services.chat_service import ChatService
from app.llm.groq_client import generate_response
from app.whatsapp.client import send_agent_button
from app.whatsapp.client import send_text_message
from app.whatsapp.client import send_main_menu
from app.conversation.conversation_state import ConversationStateManager
from app.whatsapp.webhook import router as whatsapp_router
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

app.include_router(whatsapp_router)

chat_service = ChatService()
state_manager = ConversationStateManager()


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    customer_phone: str | None = None
    conversation_id: str | None = None


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

    return chat_service.process_message(
        message=request.message,
        customer_phone=request.customer_phone,
        conversation_id=request.conversation_id
    )
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

#==========================================================
# Test: Memory Context Builder
#==========================================================
@app.post("/test/memory")
def test_memory(request: ChatRequest):

    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory()

    conversation_id = (
        request.conversation_id
        or "memory_test_001"
    )

    memory.save_message(
        conversation_id=conversation_id,
        role="user",
        message=request.message,
        channel="api"
    )

    messages = memory.get_recent_messages(
        conversation_id
    )

    return {
        "conversation_id": conversation_id,
        "messages": messages
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

#============================================================
# TEST: CLASSIFIER
#============================================================   

@app.post("/test/classifier")
def test_classifier(request: ChatRequest):

    prediction = chat_service.classifier.predict(
        request.message
    )

    decision = chat_service.confidence_checker.evaluate(
        prediction
    )

    return {
        "message": request.message,
        "prediction": prediction,
        "decision": decision
    }


# =========================================================
# ROUTER TEST
# =========================================================

@app.post("/route")
def route_message(req: ChatRequest):

    pred = chat_service.classifier.predict(req.message)

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

        out["retrieval"] = chat_service.retriever.search(
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
        "results": chat_service.retriever.search(
        req.message,
        5
    )
    }

#========================================================
#Test: Send WhatsApp Message
#========================================================
@app.post("/test/whatsapp")
def test_whatsapp():

    result = send_text_message(
        recipient_phone="919360343392",
        message="🤖 Hello from Cookd AI! WhatsApp connection is working."
    )

    return {
        "success": True,
        "whatsapp_response": result
    }

#=============================================================
#Test: Send WhatsApp Agent Button
#=============================================================
@app.post("/test/whatsapp-agent-button")
def test_whatsapp_agent_button():
    result = send_agent_button(
        recipient_phone="919360343392"
    )

    return {
        "success": True,
        "whatsapp_response": result
    }
#=========================================================
#Test: Send WhatsApp Main Menu
#=========================================================
@app.post("/test/whatsapp-menu")
def test_whatsapp_menu():

    result = send_main_menu(
        recipient_phone="919360343392"
    )

    return {
        "success": True,
        "whatsapp_response": result
    }

#=========================================================
#Test: Conversation State Manager
#=========================================================
@app.get("/test/conversation-state")
def test_conversation_state():

    conversation_id = "test:123"

    session = state_manager.get_or_create_session(
        conversation_id=conversation_id
    )

    return {
        "success": True,
        "session": session,
    }

@app.get("/test/conversation-state/track")
def test_track_state():

    result = state_manager.update_state(
        conversation_id="test:123",
        state="track_orders",
        current_action="track_orders",
    )

    return {
        "success": True,
        "result": result,
    }

@app.get("/test/conversation-state/reset")
def test_reset_conversation_state():

    result = state_manager.reset_session(
        conversation_id="test:123"
    )

    return {
        "success": True,
        "message": "Session reset",
        "result": result,
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