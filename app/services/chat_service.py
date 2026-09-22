from app.classifier.predict import IntentClassifier
from app.rag.retriever import CookdRetriever
from app.llm.groq_client import generate_response
from app.classifier.confidence import ConfidenceDecision
from app.tools.supabase_tools import (
    track_latest_order_by_phone,
)


class ChatService:

    def __init__(self):
        self.classifier = IntentClassifier()
        self.retriever = CookdRetriever()
        self.confidence_checker = ConfidenceDecision()

    # -----------------------------------------------------
    # RAG CONTEXT BUILDER
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # MAIN CHAT PROCESSOR
    # -----------------------------------------------------

    def process_message(
        self,
        message: str,
        customer_phone: str | None = None
    ):

        # -------------------------------------------------
        # 1. CLASSIFY
        # -------------------------------------------------

        prediction = self.classifier.predict(message)

        intent = prediction["intent"]
        confidence = prediction["confidence"]
        decision = self.confidence_checker.evaluate(prediction)
        if decision["action"] == "clarify":

            return {
                "success": True,
                "intent": intent,
                "confidence": confidence,
                "action": "clarify",
                "reason": decision["reason"],
                "answer": (
                    "I can help with products, recipes, "
                    "orders, recommendations, or support. "
                    "What would you like help with? 😊"
                )
            }

        # -------------------------------------------------
        # 2. ORDER TRACKING
        # -------------------------------------------------

        if intent == "order_tracking":

            if not customer_phone:

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

            result = track_latest_order_by_phone(
                customer_phone
            )

            if not result["success"]:

                return {
                    "success": False,
                    "intent": intent,
                    "confidence": confidence,
                    "error": result["error"]
                }

            customer = result["customer"]
            order = result["order"]

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

            answer = generate_response(
                user_message=message,
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


        # -------------------------------------------------
        # 3. PRODUCT QUESTION
        # -------------------------------------------------

        if intent == "product_question":

            results = self.retriever.search(
                message,
                top_k=5,
                doc_type="product"
            )

            context = self.build_rag_context(results)

            answer = generate_response(
                user_message=message,
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


        # -------------------------------------------------
        # 4. RECIPE FINDING
        # -------------------------------------------------

        if intent == "recipe_finding":

            results = self.retriever.search(
                message,
                top_k=5,
                doc_type="recipe"
            )

            context = self.build_rag_context(results)

            answer = generate_response(
                user_message=message,
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


        # -------------------------------------------------
        # 5. GENERAL FAQ
        # -------------------------------------------------

        if intent == "general_faq":

            results = self.retriever.search(
                message,
                top_k=5,
                doc_type="faq"
            )

            context = self.build_rag_context(results)

            answer = generate_response(
                user_message=message,
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


        # -------------------------------------------------
        # 6. RECOMMENDATION
        # -------------------------------------------------

        if intent == "recommendation":

            results = self.retriever.search(
                message,
                top_k=5
            )

            context = self.build_rag_context(results)

            answer = generate_response(
                user_message=message,
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


        # -------------------------------------------------
        # 7. FALLBACK
        # -------------------------------------------------

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