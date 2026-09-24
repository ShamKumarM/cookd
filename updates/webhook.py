from fastapi import APIRouter, Request, Query
from fastapi.responses import PlainTextResponse

import os

from app.services.chat_service import ChatService
from app.whatsapp.client import (
    send_text_message,
    send_main_menu,
    send_resolution_buttons,
)
from app.whatsapp.button_router import WhatsAppButtonRouter
from app.conversation.conversation_state import (
    ConversationStateManager,
)


router = APIRouter()

chat_service = ChatService()
button_router = WhatsAppButtonRouter()
state_manager = ConversationStateManager()


WHATSAPP_VERIFY_TOKEN = os.getenv(
    "WHATSAPP_VERIFY_TOKEN",
    "cookd_whatsapp_dev_2026",
)


@router.get("/webhooks/whatsapp")
def verify_whatsapp_webhook(
    hub_mode: str = Query(
        None,
        alias="hub.mode",
    ),
    hub_verify_token: str = Query(
        None,
        alias="hub.verify_token",
    ),
    hub_challenge: str = Query(
        None,
        alias="hub.challenge",
    ),
):
    if (
        hub_mode == "subscribe"
        and hub_verify_token == WHATSAPP_VERIFY_TOKEN
    ):
        return PlainTextResponse(
            content=hub_challenge or "",
            status_code=200,
        )

    return PlainTextResponse(
        content="Verification failed",
        status_code=403,
    )


@router.post("/webhooks/whatsapp")
async def receive_whatsapp_webhook(
    request: Request,
):
    body = await request.json()

    print("\n========== WHATSAPP WEBHOOK ==========")

    try:
        # =====================================================
        # PARSE META WEBHOOK
        # =====================================================

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

        print("Sender:", sender_phone)
        print("Message type:", message_type)

        if not sender_phone:
            return {
                "success": True,
                "event": "missing_sender",
            }

        # =====================================================
        # GET ACTIVE WHATSAPP SESSION
        # =====================================================
        #
        # One active session per WhatsApp phone number.
        # Exit/expiry creates a fresh conversation_id.
        # =====================================================

        session = (
            state_manager
            .get_or_create_whatsapp_session(
                whatsapp_phone=sender_phone,
            )
        )

        conversation_id = session["conversation_id"]

        print("Conversation ID:", conversation_id)
        print("Conversation session:")
        print(session)

        # =====================================================
        # INTERACTIVE MESSAGE
        # =====================================================

        if message_type == "interactive":
            interactive = message.get(
                "interactive",
                {},
            )

            interactive_type = interactive.get(
                "type"
            )

            print(
                "Interactive type:",
                interactive_type,
            )

            # =================================================
            # MAIN MENU LIST SELECTION
            # =================================================

            if interactive_type == "list_reply":
                list_reply = interactive.get(
                    "list_reply",
                    {},
                )

                button_id = list_reply.get("id")
                button_title = list_reply.get("title")

                print("List button ID:", button_id)
                print(
                    "List button title:",
                    button_title,
                )

                result = button_router.handle_button(
                    conversation_id=conversation_id,
                    customer_phone=sender_phone,
                    button_id=button_id,
                )

                print("Button router result:")
                print(result)

                return {
                    "success": True,
                    "event": "list_button_processed",
                    "button_id": button_id,
                    "result": result,
                }

            # =================================================
            # REPLY BUTTON
            # =================================================

            elif interactive_type == "button_reply":
                button_reply = interactive.get(
                    "button_reply",
                    {},
                )

                button_id = button_reply.get("id")
                button_title = button_reply.get("title")

                print("Reply button ID:", button_id)
                print(
                    "Reply button title:",
                    button_title,
                )

                result = button_router.handle_button(
                    conversation_id=conversation_id,
                    customer_phone=sender_phone,
                    button_id=button_id,
                )

                print("Button router result:")
                print(result)

                return {
                    "success": True,
                    "event": "reply_button_processed",
                    "button_id": button_id,
                    "result": result,
                }

            else:
                print(
                    "Unsupported interactive type:",
                    interactive_type,
                )

                return {
                    "success": True,
                    "event": "unsupported_interactive",
                }

        # =====================================================
        # NORMAL TEXT MESSAGE
        # =====================================================

        elif message_type == "text":
            message_text = (
                message
                .get("text", {})
                .get("body", "")
                .strip()
            )

            print("Message:", message_text)

            # Re-read the exact current session.
            session = state_manager.get_session(
                conversation_id
            )

            # Extremely defensive fallback.
            if not session:
                session = (
                    state_manager
                    .get_or_create_whatsapp_session(
                        whatsapp_phone=sender_phone,
                    )
                )
                conversation_id = session[
                    "conversation_id"
                ]

            current_action = session.get(
                "current_action"
            )

            current_state = session.get(
                "state"
            )

            print("Current state:", current_state)
            print("Current action:", current_action)

            # -------------------------------------------------
            # No selected capability
            # -------------------------------------------------
            #
            # Fresh conversation / after expiry:
            # show the main menu.
            # -------------------------------------------------

            if not current_action:
                print(
                    "No active action. "
                    "Showing Cookd main menu."
                )

                whatsapp_result = send_main_menu(
                    recipient_phone=sender_phone
                )

                print("Main menu send result:")
                print(whatsapp_result)

                return {
                    "success": True,
                    "event": "main_menu_sent",
                }

            # -------------------------------------------------
            # Process according to selected capability
            # -------------------------------------------------

            result = chat_service.process_whatsapp_message(
                message=message_text,
                action=current_action,
                customer_phone=sender_phone,
                conversation_id=conversation_id,
                channel="whatsapp",
            )

            print(
                "WhatsApp state-based result:"
            )
            print(result)

            answer = result.get("answer")

            # -------------------------------------------------
            # Send AI response
            # -------------------------------------------------

            if answer:
                whatsapp_result = send_text_message(
                    recipient_phone=sender_phone,
                    message=answer,
                )

                print("WhatsApp send result:")
                print(whatsapp_result)

            # -------------------------------------------------
            # Show resolution options ONLY when resolved
            # -------------------------------------------------

            if result.get("resolved") is True:
                resolution_result = (
                    send_resolution_buttons(
                        recipient_phone=sender_phone
                    )
                )

                print(
                    "Resolution buttons result:"
                )
                print(resolution_result)

            return {
                "success": True,
                "event": "state_based_text_processed",
                "action": current_action,
                "resolved": result.get(
                    "resolved",
                    False,
                ),
            }

        # =====================================================
        # OTHER MESSAGE TYPES
        # =====================================================

        else:
            print(
                "Unsupported message type:",
                message_type,
            )

            return {
                "success": True,
                "event": "unsupported_message_type",
                "message_type": message_type,
            }

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as e:
        print(
            "Webhook parsing error:",
            e,
        )

        return {
            "success": False,
            "error": "webhook_parsing_error",
        }

    except Exception as e:
        print(
            "Webhook processing error:",
            e,
        )

        return {
            "success": False,
            "error": "webhook_processing_error",
        }

    finally:
        print(
            "======================================\n"
        )
