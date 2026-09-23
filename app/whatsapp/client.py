import os

import httpx


WHATSAPP_ACCESS_TOKEN = os.getenv(
    "WHATSAPP_ACCESS_TOKEN"
)

WHATSAPP_PHONE_NUMBER_ID = os.getenv(
    "WHATSAPP_PHONE_NUMBER_ID"
)

WHATSAPP_GRAPH_API_VERSION = os.getenv(
    "WHATSAPP_GRAPH_API_VERSION",
    "v25.0"
)


def send_text_message(
    recipient_phone: str,
    message: str
):

    if not WHATSAPP_ACCESS_TOKEN:
        raise RuntimeError(
            "WHATSAPP_ACCESS_TOKEN is not configured"
        )

    if not WHATSAPP_PHONE_NUMBER_ID:
        raise RuntimeError(
            "WHATSAPP_PHONE_NUMBER_ID is not configured"
        )

    url = (
        f"https://graph.facebook.com/"
        f"{WHATSAPP_GRAPH_API_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )

    headers = {
        "Authorization": (
            f"Bearer {WHATSAPP_ACCESS_TOKEN}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {
            "body": message
        }
    }

    response = httpx.post(
        url,
        headers=headers,
        json=payload,
        timeout=30.0
    )

    print("WhatsApp HTTP status:", response.status_code)
    print("WhatsApp response:", response.text)

    response.raise_for_status()

    return response.json()