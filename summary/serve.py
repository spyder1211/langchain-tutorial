from langchain import hub
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

prompt = hub.pull("summary-of-city-council-committee-meeting-minutes")
llm = ChatBedrock(
    region_name="us-east-1",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
)

chain = prompt | llm | StrOutputParser()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/summary",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)