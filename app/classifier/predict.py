import pickle
from pathlib import Path
class IntentClassifier:
    def __init__(self):
        root=Path(__file__).resolve().parents[2]
        with open(root/"models/intent_classifier.pkl","rb") as f:self.model=pickle.load(f)
    def predict(self,text):
        p=self.model.predict_proba([text])[0]
        pairs=sorted(zip(self.model.classes_,p),key=lambda x:x[1],reverse=True)
        return {"intent":pairs[0][0],"confidence":float(pairs[0][1]),
                "top_intents":[{"intent":a,"confidence":float(b)} for a,b in pairs[:3]]}
