from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel

vectorstore = FAISS.from_texts(
    ["山梨県都留市出身。システムエンジニアとしてSony Ericssonでフィーチャーフォンの開発、日立製作所で大規模システム開発を経験。その後ベンチャー企業の開発部マネージャーとしてWebサービス開発、新人エンジニアの教育・マネジメントを実施。故郷である都留市でのプログラミング人材育成とデジタル事業開発にジョイン"], embedding=OpenAIEmbeddings(),
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
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = retrieval_chain.invoke("Where do you work? and what is your job? please answer in Japanese.")
print(output)