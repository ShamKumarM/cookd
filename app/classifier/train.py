import json,pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

root=Path(__file__).resolve().parents[2]
rows=json.loads((root/"data/intent_training.json").read_text())
model=Pipeline([
 ("tfidf",TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True)),
 ("clf",LogisticRegression(max_iter=2000,class_weight="balanced"))
])
model.fit([x["text"] for x in rows],[x["intent"] for x in rows])
(root/"models").mkdir(exist_ok=True)
with open(root/"models/intent_classifier.pkl","wb") as f: pickle.dump(model,f)
print("trained",len(rows),"examples")
