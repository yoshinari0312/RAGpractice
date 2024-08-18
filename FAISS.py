from openai import OpenAI
import numpy as np
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Qdrant
from langchain.schema import Document

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings()


def ask_question(question, context):
    """ GPT-4を使って質問に答えます """
    messages = [
        {"role": "system", "content": "あなたは人間です。"},
        {"role": "user", "content": f"以下の質問に以下の情報をベースにして100文字以内で答えてください。\n\n[ユーザーの質問]\n{question}\n\n[情報]\n{context}"}
    ]
    print(messages[1]["content"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 情報
    manuals = [
        "Q: auIDでログインするにはどうすればいいですか？ A: 'https://connect.auone.jp/net/vwc/cca_lg_eu_nets/login?targeturl=https%3A%2F%2Fid.auone.jp%2Findex.html%3Fstate%3Dlogin'に飛んでください。",
        "Q: スマホの通信が遅くなりました。なぜでしょうか？ A: 通信量の上限に達したかもしれません。翌月まで待ってください。",
        "Q: 私のauIDが知りたいです。 A: あなたのauIDはyoshinari0312です。",
        "Q: 請求書はどこで確認できますか？ A: 'https://my.au.com/maintenance'にログインして請求書を確認できます。",
        "Q: SIMカードの交換方法を教えてください。 A: 新しいSIMカードを挿入し、再起動してください。詳細はauのサポートページをご覧ください。",
        "Q: 電話が突然つながらなくなりました。どうすればいいですか？ A: 機内モードがオンになっていないか確認し、再起動してみてください。",
        "Q: 海外でスマホを使いたいのですが、何か設定が必要ですか？ A: 海外ローミングを有効にするために、設定アプリの「モバイルデータ通信」から「データローミング」をオンにしてください。",
        "Q: メールが届かないのですが、どうすればいいですか？ A: 迷惑メールフィルターを確認し、受信設定を見直してください。",
        "Q: 通話履歴を確認したいです。どこで見られますか？ A: 'https://my.au.com/call-history'にログインして通話履歴を確認できます。",
        "Q: auポイントの使い方を教えてください。 A: auポイントは、オンラインショップでの購入や請求金額の支払いに使用できます。詳細はauの公式サイトをご覧ください。",
        "Q: 契約内容を変更したいのですが、どうすればいいですか？ A: 'https://my.au.com/account'にログインし、契約内容を変更できます。",
        "Q: 新しいプランに変更したいです。どのように手続きをすればいいですか？ A: 'https://my.au.com/plan-change'にアクセスし、画面の指示に従って手続きを行ってください。",
        "Q: 端末のアップデートが失敗しました。どうすればいいですか？ A: 再起動してから再度アップデートを試みてください。それでも解決しない場合は、auショップにお持ちください。",
        "Q: 紛失したスマホを見つける方法はありますか？ A: 'Find My Device' や 'Find My iPhone' を使用して、紛失したスマホの場所を確認できます。",
        "Q: パスワードを忘れてしまいました。どうすればいいですか？ A: パスワードリセットページ 'https://my.au.com/reset-password' でパスワードを再設定できます。",
        "Q: 契約を解約したいのですが、どのように手続きをすればいいですか？ A: 'https://my.au.com/cancel'にアクセスし、解約手続きを行ってください。",
        "Q: データ通信を節約する方法を教えてください。 A: Wi-Fiを積極的に利用し、バックグラウンドでのデータ使用を制限することで、データ通信を節約できます。",
        "Q: 利用明細書を郵送で受け取りたいです。どうすればいいですか？ A: 'https://my.au.com/preferences'にアクセスして、郵送の設定を行ってください。",
        "Q: 家族割を適用したいのですが、どうすればいいですか？ A: 'https://my.au.com/family-discount' から家族割の申請が可能です。",
        "Q: スマホが動かなくなりました。どうすればいいですか？ A: 電源ボタンと音量ボタンを同時に長押しして、強制的に再起動してください。それでも解決しない場合は、auショップにお持ちください。"
    ]
    # 文書をDocumentオブジェクトに変換
    documents = [Document(page_content=manual) for manual in manuals]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)
    db = FAISS.from_documents(split_documents, embeddings)

    # 質問
    question = input("質問を入力してください: ")

    docs = db.similarity_search(question)

    # GPTモデルに質問
    answer = ask_question(question, docs[0].page_content)
    print(answer)
