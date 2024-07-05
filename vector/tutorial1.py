from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock,BedrockEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

documents = [
    Document(
        page_content="織田信長は日本の戦国時代において革新的な武将でした。彼は新しい戦術や技術を導入し、日本の統一を進める大きな力となりました。彼の大胆な戦略は多くの敵を驚かせました。",
        metadata={"source": "https://example.com/nobunaga"},
    ),
    Document(
        page_content="豊臣秀吉は農民から身を起こし、最終的には日本の最高権力者となりました。彼は多くの城を築き、国内の秩序を維持するために努力しました。彼の政治手腕は高く評価されています。",
        metadata={"source": "https://example.com/hideyoshi"},
    ),
    Document(
        page_content="徳川家康は関ヶ原の戦いで勝利を収め、江戸幕府を開きました。彼の治世は平和と安定をもたらし、約260年間続く江戸時代の基盤を築きました。彼の統治方法は後世に多大な影響を与えました。",
        metadata={"source": "https://example.com/ieyasu"},
    ),
    Document(
        page_content="石田三成は豊臣政権の重要な官僚であり、関ヶ原の戦いで西軍の指導者として知られています。彼は律儀で誠実な性格で、多くの忠臣を持ちましたが、最終的には徳川家康に敗北しました。",
        metadata={"source": "https://example.com/mitsunari"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=BedrockEmbeddings(),
)

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

#result = vectorstore.similarity_search_with_score("徳川家康")
# 抽出されたドキュメントとスコアを表示する例
# for doc, score in result:
#     print(f"Score: {score}")
#     print(f"Page Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

message = """
Answer this question using the provided context only.

{question}

<context>
{context}
</context>
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

ragchain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()

response = ragchain.invoke("織田信長はどのような武将でしたか？")
print(response)