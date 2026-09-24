from app.conversation.conversation_state import (
    ConversationStateManager,
)
from app.whatsapp.client import (
    send_text_message,
    send_main_menu,
    send_action_prompt,
)


class WhatsAppButtonRouter:

    def __init__(self):
        self.state_manager = ConversationStateManager()

    def handle_button(
        self,
        conversation_id: str,
        customer_phone: str,
        button_id: str,
        customer_id=None,
    ):
        print(f"Button clicked: {button_id}")

        # =====================================================
        # BACK TO MAIN MENU
        # =====================================================

        if button_id == "back_to_main_menu":
            self.state_manager.start_main_menu(
                conversation_id=conversation_id,
                customer_id=customer_id,
            )

            send_main_menu(
                recipient_phone=customer_phone
            )

            return {
                "success": True,
                "action": "main_menu",
                "message": "Returned to main menu",
            }

        # =====================================================
        # SAME QUERY HELP
        # =====================================================

        if button_id == "same_query_help":
            session = self.state_manager.get_session(
                conversation_id
            )

            if not session:
                send_main_menu(
                    recipient_phone=customer_phone
                )

                return {
                    "success": True,
                    "action": "main_menu",
                    "message": (
                        "Session not found. "
                        "Showing main menu."
                    ),
                }

            current_action = session.get(
                "current_action"
            )

            if not current_action:
                send_main_menu(
                    recipient_phone=customer_phone
                )

                return {
                    "success": True,
                    "action": "main_menu",
                    "message": (
                        "No active action. "
                        "Showing main menu."
                    ),
                }

            # Keep the same capability active.
            # Memory is NOT deleted.
            self.state_manager.update_state(
                conversation_id=conversation_id,
                state=current_action,
                current_action=current_action,
            )

            send_action_prompt(
                recipient_phone=customer_phone,
                action=current_action,
            )

            return {
                "success": True,
                "action": "same_query_help",
                "current_action": current_action,
                "message": (
                    "Continuing with the same query."
                ),
            }

        # =====================================================
        # ANOTHER OPTION
        # =====================================================

        if button_id == "another_option":
            # Important: update_state() now supports
            # current_action=None and actually clears it.
            self.state_manager.update_state(
                conversation_id=conversation_id,
                state="main_menu",
                current_action=None,
            )

            send_main_menu(
                recipient_phone=customer_phone
            )

            return {
                "success": True,
                "action": "main_menu",
                "message": "Showing main menu.",
            }

        # =====================================================
        # EXIT CONVERSATION
        # =====================================================

        if button_id == "exit_conversation":
            # Deletes only the active session.
            # conversation_events remain untouched.
            self.state_manager.end_conversation(
                conversation_id=conversation_id
            )

            send_text_message(
                recipient_phone=customer_phone,
                message=(
                    "Your conversation has been ended. "
                    "Thank you for chatting with Cookd! 😊\n\n"
                    "Send *Hi* anytime to start a fresh "
                    "conversation."
                ),
            )

            return {
                "success": True,
                "action": "exit_conversation",
                "message": "Conversation ended.",
            }

        # =====================================================
        # MAIN MENU ACTIONS
        # =====================================================

        valid_actions = {
            "orders_details": "orders_details",
            "track_orders": "track_orders",
            "product_queries": "product_queries",
            "recipe_guide": "recipe_guide",
            "cookd_finder": "cookd_finder",
        }

        if button_id in valid_actions:
            action = valid_actions[button_id]

            self.state_manager.update_state(
                conversation_id=conversation_id,
                state=action,
                current_action=action,
            )

            send_action_prompt(
                recipient_phone=customer_phone,
                action=action,
            )

            return {
                "success": True,
                "action": action,
                "message": "Action selected",
            }

        # =====================================================
        # HUMAN SUPPORT
        # =====================================================

        if button_id == "human_support":
            self.state_manager.update_state(
                conversation_id=conversation_id,
                state="human_support",
                current_action="human_support",
            )

            return {
                "success": True,
                "action": "human_support",
                "message": "Human support requested",
            }

        # =====================================================
        # UNKNOWN BUTTON
        # =====================================================

        return {
            "success": False,
            "action": "unknown",
            "message": "Unknown button",
        }
