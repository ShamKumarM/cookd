from app.rag.retriever import CookdRetriever
r=CookdRetriever()
for q in ["I want a biryani for three people","Which Cookd product is good for fish fry?","What is Madras 65?","I have chicken and want a recipe"]:
    print("\nQUERY:",q)
    for x in r.search(q,3):
        print(x["metadata"],"distance=",x["distance"])
        print(x["text"][:220].replace("\n"," "))
