from fastapi import APIRouter, Request, Query
from fastapi.responses import PlainTextResponse
from app.services.chat_service import ChatService
from app.whatsapp.client import send_text_message
import os

router = APIRouter()

WHATSAPP_VERIFY_TOKEN = os.getenv(
    "WHATSAPP_VERIFY_TOKEN",
    "cookd_whatsapp_dev_2026"
)

chat_service = ChatService()

@router.get("/webhooks/whatsapp")
def verify_whatsapp_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    if (
        hub_mode == "subscribe"
        and hub_verify_token == WHATSAPP_VERIFY_TOKEN
    ):
        return PlainTextResponse(
            content=hub_challenge or "",
            status_code=200
        )

    return PlainTextResponse(
        content="Verification failed",
        status_code=403
    )


@router.post("/webhooks/whatsapp")
async def receive_whatsapp_webhook(request: Request):
    body = await request.json()

    print("\n========== WHATSAPP WEBHOOK ==========")

    try:
        entry = body["entry"][0]
        change = entry["changes"][0]
        value = change["value"]

        messages = value.get("messages", [])

        if not messages:
            print("No messages in webhook event.")
            return {"success": True}

        message = messages[0]

        sender_phone = message.get("from")
        message_type = message.get("type")

        if message_type != "text":
            print("Unsupported message type:", message_type)
            return {"success": True}

        message_text = message["text"]["body"]

        print("Sender:", sender_phone)
        print("Message:", message_text)

        # Stable conversation ID for this WhatsApp user
        conversation_id = f"whatsapp:{sender_phone}"

        # Send message through existing Cookd AI pipeline
        result = chat_service.process_message(
            message=message_text,
            customer_phone=sender_phone,
            conversation_id=conversation_id,
            channel="whatsapp",
        )

        print("ChatService result:")
        print(result)
        answer = result.get("answer")

        if answer:
            whatsapp_result = send_text_message(
                recipient_phone=sender_phone,
                message=answer,
            )

            print("WhatsApp send result:")
            print(whatsapp_result)

    except (KeyError, IndexError, TypeError) as e:
        print("Webhook parsing error:", e)

    except Exception as e:
        print("ChatService error:", e)

    print("======================================\n")

    return {"success": True}