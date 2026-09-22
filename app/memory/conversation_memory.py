from typing import Optional
import uuid

from app.tools.supabase_tools import supabase


class ConversationMemory:

    def __init__(self, max_messages: int = 8):
        self.max_messages = max_messages

    # =====================================================
    # SAVE MESSAGE
    # =====================================================

    def save_message(
        self,
        conversation_id: str,
        role: str,
        message: str,
        customer_id: Optional[str] = None,
        intent: Optional[str] = None,
        channel: str = "api",
        ai_resolved: Optional[bool] = None,
    ):

        data = {
            # Existing table requires this
            "id": f"evt_{uuid.uuid4().hex[:12]}",

            "conversation_id": conversation_id,
            "role": role,
            "message": message,
            "channel": channel,
        }

        if customer_id:
            data["customer_id"] = customer_id

        if intent:
            data["intent"] = intent

        if ai_resolved is not None:
            data["ai_resolved"] = ai_resolved

        response = (
            supabase
            .table("conversation_events")
            .insert(data)
            .execute()
        )

        return response.data

    # =====================================================
    # GET RECENT MESSAGES
    # =====================================================

    def get_recent_messages(
        self,
        conversation_id: str
    ):

        response = (
            supabase
            .table("conversation_events")
            .select(
                "id, conversation_id, role, message, intent, created_at"
            )
            .eq(
                "conversation_id",
                conversation_id
            )
            .not_.is_("message", "null")
            .order(
                "created_at",
                desc=True
            )
            .limit(self.max_messages)
            .execute()
        )

        messages = response.data or []

        # Newest → oldest from database.
        # Reverse so the LLM sees chronological order.

        messages.reverse()

        return messages

    # =====================================================
    # BUILD MEMORY CONTEXT
    # =====================================================

    def build_context(
        self,
        conversation_id: str
    ):

        messages = self.get_recent_messages(
            conversation_id
        )

        if not messages:
            return "No previous conversation history."

        context_parts = []

        for item in messages:

            role = item.get("role", "")
            content = item.get("message", "")

            if role == "user":
                label = "USER"

            elif role == "assistant":
                label = "ASSISTANT"

            else:
                label = role.upper()

            context_parts.append(
                f"{label}: {content}"
            )

        return "\n".join(context_parts)