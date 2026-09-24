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
    "v25.0",
)


def _get_url():
    if not WHATSAPP_ACCESS_TOKEN:
        raise RuntimeError(
            "WHATSAPP_ACCESS_TOKEN is not configured"
        )

    if not WHATSAPP_PHONE_NUMBER_ID:
        raise RuntimeError(
            "WHATSAPP_PHONE_NUMBER_ID is not configured"
        )

    return (
        f"https://graph.facebook.com/"
        f"{WHATSAPP_GRAPH_API_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )


def _post(payload):
    url = _get_url()

    headers = {
        "Authorization": (
            f"Bearer {WHATSAPP_ACCESS_TOKEN}"
        ),
        "Content-Type": "application/json",
    }

    response = httpx.post(
        url,
        headers=headers,
        json=payload,
        timeout=30.0,
    )

    print(
        "WhatsApp HTTP status:",
        response.status_code,
    )
    print(
        "WhatsApp response:",
        response.text,
    )

    response.raise_for_status()
    return response.json()


def send_text_message(
    recipient_phone: str,
    message: str,
):
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {
            "body": message,
        },
    }

    return _post(payload)


def send_agent_button(
    recipient_phone: str,
    message: str = (
        "Would you like to connect with a support agent?"
    ),
):
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": message,
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "connect_with_agent",
                            "title": "Connect with Agent",
                        },
                    }
                ],
            },
        },
    }

    return _post(payload)


def send_main_menu(
    recipient_phone: str,
    message: str = (
        "👋 Hi! Welcome to Cookd AI.\n\n"
        "How can I help you today?"
    ),
):
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {
                "text": message,
            },
            "action": {
                "button": "Choose an option",
                "sections": [
                    {
                        "title": "Cookd Support",
                        "rows": [
                            {
                                "id": "orders_details",
                                "title": "Orders Details",
                                "description": (
                                    "View your previous orders"
                                ),
                            },
                            {
                                "id": "track_orders",
                                "title": "Track Orders",
                                "description": (
                                    "Check your latest order status"
                                ),
                            },
                            {
                                "id": "product_queries",
                                "title": "Product Queries",
                                "description": (
                                    "Ask about Cookd products"
                                ),
                            },
                            {
                                "id": "recipe_guide",
                                "title": "Recipe Guide",
                                "description": (
                                    "Get recipe and cooking guidance"
                                ),
                            },
                            {
                                "id": "cookd_finder",
                                "title": "Cookd Finder",
                                "description": (
                                    "Find Cookd products near you"
                                ),
                            },
                            {
                                "id": "human_support",
                                "title": "Others",
                                "description": (
                                    "Connect with customer support"
                                ),
                            },
                        ],
                    }
                ],
            },
        },
    }

    return _post(payload)


def send_action_prompt(
    recipient_phone: str,
    action: str,
):
    prompts = {
        "orders_details": (
            "📦 *Orders Details selected.*\n\n"
            "You can now type your order-related request.\n\n"
            "For example:\n"
            "• Show my previous orders\n"
            "• Show my order details\n"
            "• What did I order?"
        ),
        "track_orders": (
            "🚚 *Track Orders selected.*\n\n"
            "You can now type your order tracking question.\n\n"
            "For example:\n"
            "• Where is my order?\n"
            "• What is my order status?\n"
            "• Track my latest order"
        ),
        "product_queries": (
            "🌶️ *Product Queries selected.*\n\n"
            "You can now type your product-related question.\n\n"
            "For example:\n"
            "• Tell me about Madras 65\n"
            "• Which masala is good for chicken?\n"
            "• What products do you have?"
        ),
        "recipe_guide": (
            "🍳 *Recipe Guide selected.*\n\n"
            "You can now type your recipe or cooking question.\n\n"
            "For example:\n"
            "• How do I make chicken biryani?\n"
            "• Give me a recipe using Madras 65\n"
            "• How should I cook this masala?"
        ),
        "cookd_finder": (
            "📍 *Cookd Finder selected.*\n\n"
            "Cookd Finder is currently under development.\n"
            "We'll help you find Cookd products near you soon! 😊"
        ),
    }

    message = prompts.get(
        action,
        "Please type your question and I'll help you.",
    )

    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": message,
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "back_to_main_menu",
                            "title": "Back to Main Menu",
                        },
                    }
                ],
            },
        },
    }

    return _post(payload)


def send_resolution_buttons(recipient_phone: str):
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": (
                    "Is there anything else I can help "
                    "you with?"
                ),
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "same_query_help",
                            "title": "Same query help",
                        },
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "another_option",
                            "title": "Another option",
                        },
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "exit_conversation",
                            "title": "Exit conversation",
                        },
                    },
                ],
            },
        },
    }

    return _post(payload)
