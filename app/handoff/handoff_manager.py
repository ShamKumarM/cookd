class HandoffManager:

    HANDOFF_PHRASES = [
        "talk to human",
        "talk to a human",
        "human agent",
        "customer support",
        "speak to someone",
        "speak to a human",
        "connect me to an agent",
        "connect me with an agent",
        "real person",
        "live agent",
        "customer care",
    ]

    def should_handoff(self, message: str) -> bool:

        text = message.lower().strip()

        return any(
            phrase in text
            for phrase in self.HANDOFF_PHRASES
        )