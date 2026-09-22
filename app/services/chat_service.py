import uuid

from app.classifier.predict import IntentClassifier
from app.classifier.confidence import ConfidenceDecision
from app.rag.retriever import CookdRetriever
from app.llm.groq_client import generate_response
from app.memory.conversation_memory import ConversationMemory
from app.tools.supabase_tools import (
    track_latest_order_by_phone,
)


class ChatService:

    def __init__(self):

        self.classifier = IntentClassifier()
        self.retriever = CookdRetriever()
        self.confidence_checker = ConfidenceDecision()
        self.memory = ConversationMemory()

    # =====================================================
    # RAG CONTEXT BUILDER
    # =====================================================

    def build_rag_context(self, results):

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

    # =====================================================
    # COMBINE MEMORY + VERIFIED INFORMATION
    # =====================================================

    def build_combined_context(
        self,
        memory_context,
        verified_context
    ):

        return f"""
CONVERSATION HISTORY:

{memory_context}


VERIFIED INFORMATION:

{verified_context}
"""

    # =====================================================
    # SAVE CONVERSATION EXCHANGE
    # =====================================================

    def save_exchange(
        self,
        conversation_id,
        user_message,
        assistant_message,
        intent,
        customer_id=None,
        channel="api"
    ):

        # Save user message
        self.memory.save_message(
            conversation_id=conversation_id,
            role="user",
            message=user_message,
            customer_id=customer_id,
            intent=intent,
            channel=channel
        )

        # Save assistant response
        self.memory.save_message(
            conversation_id=conversation_id,
            role="assistant",
            message=assistant_message,
            customer_id=customer_id,
            intent=intent,
            channel=channel,
            ai_resolved=True
        )

    # =====================================================
    # MAIN CHAT PROCESSOR
    # =====================================================

    def process_message(
        self,
        message: str,
        customer_phone: str | None = None,
        conversation_id: str | None = None
    ):

        # -------------------------------------------------
        # 0. CREATE CONVERSATION ID
        # -------------------------------------------------

        if not conversation_id:

            conversation_id = str(
                uuid.uuid4()
            )

        # -------------------------------------------------
        # 1. CLASSIFY USER MESSAGE
        # -------------------------------------------------

        prediction = self.classifier.predict(
            message
        )

        intent = prediction["intent"]
        confidence = prediction["confidence"]

        # -------------------------------------------------
        # 2. CONFIDENCE CHECK
        # -------------------------------------------------

        decision = self.confidence_checker.evaluate(
            prediction
        )

        # -------------------------------------------------
        # 3. LOAD PREVIOUS CONVERSATION
        # -------------------------------------------------

        memory_context = self.memory.build_context(
            conversation_id
        )

        # -------------------------------------------------
        # 4. HANDLE LOW CONFIDENCE
        # -------------------------------------------------

        if decision["action"] == "clarify":

            answer = (
                "I can help with products, recipes, "
                "orders, recommendations, or support. "
                "What would you like help with? 😊"
            )

            self.save_exchange(
                conversation_id=conversation_id,
                user_message=message,
                assistant_message=answer,
                intent=intent,
                channel="api"
            )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "intent": intent,
                "confidence": confidence,
                "action": "clarify",
                "reason": decision["reason"],
                "answer": answer
            }

        # =================================================
        # 5. ORDER TRACKING → SUPABASE
        # =================================================

        if intent == "order_tracking":

            if not customer_phone:

                answer = (
                    "Please provide your phone number "
                    "so I can check your order."
                )

                self.save_exchange(
                    conversation_id=conversation_id,
                    user_message=message,
                    assistant_message=answer,
                    intent=intent,
                    channel="api"
                )

                return {
                    "success": False,
                    "conversation_id": conversation_id,
                    "intent": intent,
                    "confidence": confidence,
                    "error": "customer_phone_required",
                    "message": answer
                }

            result = track_latest_order_by_phone(
                customer_phone
            )

            if not result["success"]:

                answer = (
                    "I couldn't find an order for that "
                    "customer information."
                )

                self.save_exchange(
                    conversation_id=conversation_id,
                    user_message=message,
                    assistant_message=answer,
                    intent=intent,
                    channel="api"
                )

                return {
                    "success": False,
                    "conversation_id": conversation_id,
                    "intent": intent,
                    "confidence": confidence,
                    "error": result["error"],
                    "message": answer
                }

            customer = result["customer"]
            order = result["order"]

            customer_id = customer.get("id")

            verified_context = f"""
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

            combined_context = self.build_combined_context(
                memory_context,
                verified_context
            )

            answer = generate_response(
                user_message=message,
                context=combined_context
            )

            self.save_exchange(
                conversation_id=conversation_id,
                user_message=message,
                assistant_message=answer,
                intent=intent,
                customer_id=customer_id,
                channel="api"
            )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "intent": intent,
                "confidence": confidence,
                "source": "supabase",
                "answer": answer,
                "order": order
            }

        # =================================================
        # 6. PRODUCT QUESTION → CHROMA
        # =================================================

        if intent == "product_question":

            results = self.retriever.search(
                message,
                top_k=5,
                doc_type="product"
            )

            verified_context = self.build_rag_context(
                results
            )

            combined_context = self.build_combined_context(
                memory_context,
                verified_context
            )

            answer = generate_response(
                user_message=message,
                context=combined_context
            )

            self.save_exchange(
                conversation_id=conversation_id,
                user_message=message,
                assistant_message=answer,
                intent=intent,
                channel="api"
            )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "intent": intent,
                "confidence": confidence,
                "source": "chroma",
                "answer": answer,
                "retrieved_results": results
            }

        # =================================================
        # 7. RECIPE FINDING → CHROMA
        # =================================================

        if intent == "recipe_finding":

            results = self.retriever.search(
                message,
                top_k=5,
                doc_type="recipe"
            )

            verified_context = self.build_rag_context(
                results
            )

            combined_context = self.build_combined_context(
                memory_context,
                verified_context
            )

            answer = generate_response(
                user_message=message,
                context=combined_context
            )

            self.save_exchange(
                conversation_id=conversation_id,
                user_message=message,
                assistant_message=answer,
                intent=intent,
                channel="api"
            )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "intent": intent,
                "confidence": confidence,
                "source": "chroma",
                "answer": answer,
                "retrieved_results": results
            }

        # =================================================
        # 8. GENERAL FAQ → CHROMA
        # =================================================

        if intent == "general_faq":

            results = self.retriever.search(
                message,
                top_k=5,
                doc_type="faq"
            )

            verified_context = self.build_rag_context(
                results
            )

            combined_context = self.build_combined_context(
                memory_context,
                verified_context
            )

            answer = generate_response(
                user_message=message,
                context=combined_context
            )

            self.save_exchange(
                conversation_id=conversation_id,
                user_message=message,
                assistant_message=answer,
                intent=intent,
                channel="api"
            )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "intent": intent,
                "confidence": confidence,
                "source": "chroma",
                "answer": answer,
                "retrieved_results": results
            }

        # =================================================
        # 9. RECOMMENDATION → CHROMA
        # =================================================

        if intent == "recommendation":

            results = self.retriever.search(
                message,
                top_k=5
            )

            verified_context = self.build_rag_context(
                results
            )

            combined_context = self.build_combined_context(
                memory_context,
                verified_context
            )

            answer = generate_response(
                user_message=message,
                context=combined_context
            )

            self.save_exchange(
                conversation_id=conversation_id,
                user_message=message,
                assistant_message=answer,
                intent=intent,
                channel="api"
            )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "intent": intent,
                "confidence": confidence,
                "source": "chroma",
                "answer": answer,
                "retrieved_results": results
            }

        # =================================================
        # 10. FALLBACK
        # =================================================

        answer = (
            "I need a little more information "
            "to help with that."
        )

        self.save_exchange(
            conversation_id=conversation_id,
            user_message=message,
            assistant_message=answer,
            intent=intent,
            channel="api"
        )

        return {
            "success": True,
            "conversation_id": conversation_id,
            "intent": intent,
            "confidence": confidence,
            "source": "unknown",
            "answer": answer
        }