from langchain_community.document_loaders import SitemapLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.pydantic_v1 import BaseModel

# loader = SitemapLoader(web_path="https://c-table.co.jp/sitemap.xml")
loader = SitemapLoader(web_path="https://furusato-nippon.com/sitemap.xml")

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 400,
    chunk_overlap = 0,
    length_function = len,
)

index = VectorstoreIndexCreator(
    vectorstore_cls=InMemoryVectorStore,
    embedding=BedrockEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

retriever = index.vectorstore.as_retriever()

## サイトマップから取得した内容を元にbedrockで日本語で回答する
template = [("system", """Answer in Japanese the question based only on the following context:

<context>
{context}
</context>
"""
), ("human", "質問：{input}")]
prompt = ChatPromptTemplate.from_messages(template)

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

# chain = retriever | prompt | model | StrOutputParser()
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(model, prompt))

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

class ChainInput(BaseModel):
    input: str

add_routes(
    app,
    # chain,
    chain.with_types(input_type=ChainInput),
    path="/sitemap",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)

