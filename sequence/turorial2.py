from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic} in japanese")

chain = prompt | model | StrOutputParser()

analysis_prompt = ChatPromptTemplate.from_template("is this joke funny? {joke}")

composed_chain_with_lambda = ( chain | (lambda input: {"joke": input}) | analysis_prompt | model | StrOutputParser() )

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    composed_chain_with_lambda,
    path="/bedrock",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

