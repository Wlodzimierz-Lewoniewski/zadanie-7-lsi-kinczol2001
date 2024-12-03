import numpy as np
import re

def latent_semantic_indexing(documents, query, k):
    preprocess = lambda text: re.sub(r'[^\w\s]', '', text.lower())
    terms = sorted(set(preprocess(term) for doc in documents for term in doc.split()))
    term_index = {term: i for i, term in enumerate(terms)}
    C = np.zeros((len(terms), len(documents)))
    for j, doc in enumerate(documents):
        for term in preprocess(doc).split():
            if term in term_index:
                C[term_index[term], j] = 1
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    V_k = Vt[:k, :]
    documents_reduced = np.dot(S_k, V_k)
    query_vector = np.zeros(len(terms))
    for term in preprocess(query).split():
        if term in term_index:
            query_vector[term_index[term]] = 1
    query_reduced = np.dot(np.dot(query_vector, U_k), np.linalg.inv(S_k))
    similarities = [
        np.dot(query_reduced, documents_reduced[:, i]) / 
        (np.linalg.norm(query_reduced) * np.linalg.norm(documents_reduced[:, i]))
        for i in range(documents_reduced.shape[1])
    ]
    return [round(sim, 2) for sim in similarities]

def main():
    n = int(input())
    documents = [input() for _ in range(n)]
    query = input()
    k = int(input())
    similarities = latent_semantic_indexing(documents, query, k)
    print(similarities)

if __name__ == "__main__":
    main()
