RAG={"product_question":"product","recipe_finding":"recipe","recommendation":"mixed","general_faq":"faq"}
TOOLS={"order_tracking","feedback_complaint","return_refund","cart_action"}

def route(intent,confidence):
    if confidence<0.60:
        return {"agent":"clarification_agent","knowledge_source":None,"requires_tool":False}
    if intent in RAG:
        return {"agent":intent+"_agent","knowledge_source":RAG[intent],"requires_tool":False}
    if intent in TOOLS:
        return {"agent":intent+"_agent","knowledge_source":None,"requires_tool":True}
    return {"agent":"general_support_agent","knowledge_source":"faq","requires_tool":False}
