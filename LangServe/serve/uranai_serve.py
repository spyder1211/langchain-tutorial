#!/usr/bin/env python
from typing import List

from fastapi import FastAPI, Request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. create prompt template
# 入力された星座によって、本日の運勢を占います。
system_template = "本日の{星座}の運勢を占います。30文字程度でまとめてください。"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{星座}')
])

# 2. Create model
# gpt4oを使用
model = ChatOpenAI(model="gpt-4o")

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
