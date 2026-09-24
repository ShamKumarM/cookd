from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

from app.tools.supabase_tools import supabase


class ConversationStateManager:

    SESSION_TIMEOUT_MINUTES = 20

    STATES = {
        "main_menu",
        "orders_details",
        "track_orders",
        "product_queries",
        "recipe_guide",
        "cookd_finder",
        "human_support",
        "resolution",
        "same_query",
    }

    _UNSET = object()

    def _now(self):
        return datetime.now(timezone.utc)

    def get_session(self, conversation_id: str):
        response = (
            supabase
            .table("conversation_sessions")
            .select("*")
            .eq("conversation_id", conversation_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        return response.data[0]

    def is_expired(self, session: Optional[dict]) -> bool:
        if not session:
            return False

        last_activity = session.get("last_activity_at")
        if not last_activity:
            return False

        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(
                last_activity.replace("Z", "+00:00")
            )

        expiry_time = (
            last_activity
            + timedelta(minutes=self.SESSION_TIMEOUT_MINUTES)
        )

        return self._now() >= expiry_time

    def create_session(
        self,
        conversation_id: str,
        customer_id: Optional[str] = None,
        whatsapp_phone: Optional[str] = None,
    ):
        now = self._now().isoformat()

        data = {
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "whatsapp_phone": whatsapp_phone,
            "state": "main_menu",
            "current_action": None,
            "last_resolved_action": None,
            "last_activity_at": now,
            "created_at": now,
            "updated_at": now,
        }

        response = (
            supabase
            .table("conversation_sessions")
            .upsert(data)
            .execute()
        )

        return response.data[0] if response.data else data

    def get_or_create_session(
        self,
        conversation_id: str,
        customer_id: Optional[str] = None,
    ):
        session = self.get_session(conversation_id)

        if not session:
            return self.create_session(
                conversation_id=conversation_id,
                customer_id=customer_id,
            )

        if self.is_expired(session):
            print(f"Session expired: {conversation_id}")

            self.reset_session(conversation_id)

            return self.create_session(
                conversation_id=conversation_id,
                customer_id=customer_id,
            )

        return session

    def update_state(
        self,
        conversation_id: str,
        state: str,
        current_action=_UNSET,
        last_resolved_action=_UNSET,
    ):
        if state not in self.STATES:
            raise ValueError(
                f"Invalid conversation state: {state}"
            )

        now = self._now().isoformat()

        data = {
            "state": state,
            "last_activity_at": now,
            "updated_at": now,
        }

        # Important:
        # Passing current_action=None explicitly clears the action.
        if current_action is not self._UNSET:
            data["current_action"] = current_action

        if last_resolved_action is not self._UNSET:
            data["last_resolved_action"] = last_resolved_action

        response = (
            supabase
            .table("conversation_sessions")
            .update(data)
            .eq("conversation_id", conversation_id)
            .execute()
        )

        return response.data

    def touch(self, conversation_id: str):
        now = self._now().isoformat()

        response = (
            supabase
            .table("conversation_sessions")
            .update({
                "last_activity_at": now,
                "updated_at": now,
            })
            .eq("conversation_id", conversation_id)
            .execute()
        )

        return response.data

    def reset_session(self, conversation_id: str):
        response = (
            supabase
            .table("conversation_sessions")
            .delete()
            .eq("conversation_id", conversation_id)
            .execute()
        )

        return response.data

    def start_main_menu(
        self,
        conversation_id: str,
        customer_id: Optional[str] = None,
    ):
        session = self.get_session(conversation_id)

        if not session:
            return self.create_session(
                conversation_id=conversation_id,
                customer_id=customer_id,
            )

        return self.update_state(
            conversation_id=conversation_id,
            state="main_menu",
            current_action=None,
        )

    def end_conversation(self, conversation_id: str):
        print(
            f"Ending conversation session: {conversation_id}"
        )

        response = (
            supabase
            .table("conversation_sessions")
            .delete()
            .eq("conversation_id", conversation_id)
            .execute()
        )

        return response.data

    def get_active_session_by_phone(
        self,
        whatsapp_phone: str,
    ):
        response = (
            supabase
            .table("conversation_sessions")
            .select("*")
            .eq("whatsapp_phone", whatsapp_phone)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        return response.data[0]

    def get_or_create_whatsapp_session(
        self,
        whatsapp_phone: str,
    ):
        session = self.get_active_session_by_phone(
            whatsapp_phone
        )

        if session:
            if self.is_expired(session):
                print(
                    "WhatsApp session expired: "
                    f"{session['conversation_id']}"
                )

                self.end_conversation(
                    session["conversation_id"]
                )
            else:
                self.touch(
                    session["conversation_id"]
                )
                return session

        conversation_id = (
            f"whatsapp:{whatsapp_phone}:"
            f"{uuid.uuid4()}"
        )

        return self.create_session(
            conversation_id=conversation_id,
            whatsapp_phone=whatsapp_phone,
        )
