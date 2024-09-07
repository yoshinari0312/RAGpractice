from openai import OpenAI
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


if __name__ == "__main__":
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    # 質問
    question = input("質問を入力してください: ")

    docs = vector_store.similarity_search_with_score(question, k=2)
    for res, score in docs:
        print(f"* [SIM={score:3f}] {res.page_content}")

    # GPTモデルに質問
    messages = [
        {"role": "system", "content": "あなたは人間です。"},
        {"role": "user", "content": f"以下の質問に以下の情報をベースにして200文字以内で答えてください。\n\n[ユーザーの質問]\n{question}\n\n[情報]\n{docs[0][0].page_content}\n{docs[1][0].page_content}"}
    ]
    print(messages[1]["content"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    print(response.choices[0].message.content)
