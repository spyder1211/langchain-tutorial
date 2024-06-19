from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel
from fastapi import FastAPI, Request
from langserve import add_routes

texts = [
    "田中 太郎です。東京都出身で、現在はIT業界でソフトウェアエンジニアとして働いています。趣味は登山と読書で、休日にはよく山に登っています。これまでに10年以上のプログラミング経験があり、特にPythonとJavaに精通しています。将来はAI技術の専門家として活躍したいと考えています。",
    "鈴木 花子と申します。大阪府出身で、現在はマーケティング会社でデジタルマーケティングの専門家として働いています。趣味は料理と旅行で、週末には新しいレシピに挑戦するのが楽しみです。マーケティング業界で5年以上の経験があり、特にソーシャルメディア戦略の構築に強みがあります。",
    "佐藤 健です。北海道出身で、現在は大学で機械工学を専攻しています。趣味は自転車と映画鑑賞で、特にSF映画が好きです。大学ではロボティクスに関する研究を行っており、将来はロボット開発の分野で働きたいと考えています。",
    "山田 直美と申します。愛知県出身で、現在はファッション業界でデザイナーとして働いています。趣味は写真撮影と美術館巡りで、特に現代アートに興味があります。デザインの仕事に携わって5年以上が経ち、これまでに多くのブランドの立ち上げに貢献してきました。",
    "小林 誠です。福岡県出身で、現在はスタートアップ企業でデータサイエンティストとして働いています。趣味はランニングとチェスで、毎朝ジョギングをしています。データ解析と機械学習の分野で豊富な経験があり、特にビッグデータの分析に強みがあります。"
]

vectorstore = FAISS.from_texts(
    texts, embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4o")

retrieval_chain = (
    # RunnableParallel(context=retriever, question=RunnablePassthrough())
    # {"context": retriever, "question": RunnablePassthrough()}
    {"context": retriever, "question": {"question": RunnablePassthrough()}}
    | prompt
    | model
    | StrOutputParser()
)

# output = retrieval_chain.invoke("Where do you work? and what is your job? please answer in Japanese.")
# print(output)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    retrieval_chain,
    path="/bedrock",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

