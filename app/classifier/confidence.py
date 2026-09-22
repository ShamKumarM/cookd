class ConfidenceDecision:

    def __init__(
        self,
        min_confidence=0.25,
        ambiguity_gap=0.08
    ):
        self.min_confidence = min_confidence
        self.ambiguity_gap = ambiguity_gap


    def evaluate(self, prediction):

        confidence = prediction["confidence"]
        top_intents = prediction["top_intents"]

        # -------------------------------------------------
        # VERY LOW CONFIDENCE
        # -------------------------------------------------

        if confidence < self.min_confidence:

            return {
                "action": "clarify",
                "reason": "very_low_confidence"
            }


        # -------------------------------------------------
        # TOP-2 AMBIGUITY
        # -------------------------------------------------

        if len(top_intents) >= 2:

            second_confidence = top_intents[1]["confidence"]

            gap = confidence - second_confidence

            if gap < self.ambiguity_gap:

                return {
                    "action": "clarify",
                    "reason": "ambiguous_intent"
                }


        # -------------------------------------------------
        # OTHERWISE EXECUTE
        # -------------------------------------------------

        return {
            "action": "execute",
            "reason": "sufficient_confidence"
        }