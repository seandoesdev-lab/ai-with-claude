documents = [
    "고양이는 포유류이고 야옹하고 운다",
    "강아지는 포유류이고 멍멍하고 짖는다",
    "트랜스포머는 어텐션 기반의 신경망 구조다",
    "RAG는 검색과 생성을 결합한 기법이다",
]

def naive_search(query, docs):
    q_words = set(query.split())
    print(f"Query words: {q_words}")

    scored = []
    for i, doc in enumerate(docs):
        overlap = len(q_words & set(doc.split()))
        if overlap > 0:
            scored.append((overlap, i))
            print(f"Document {i} has {overlap} overlapping words: {set(doc.split()) & q_words}")
    
    return sorted(scored, reverse=True)


for score, idx in naive_search("RAG는 무엇인가", documents):
    print(f"Score: {score}, Document: {documents[idx]}")