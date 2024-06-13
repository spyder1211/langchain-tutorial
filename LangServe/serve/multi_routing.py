#!/usr/bin/env python
from typing import List

from fastapi import FastAPI, Request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

translate = prompt_template

# 1. create another prompt template
system_template = "与えられたキーワードに関するジョークを作成します。"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{keyword}')
])

joke = prompt_template

# 1. create another prompt template
system_template = "あなたの役割は、{コーチの特徴}を持ったコーチです。ユーザーから与えられた名前に対して、コーチの特徴に合わせた言葉遣いで叱咤激励してください。" 
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{名前}')
])

uranai = prompt_template

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
translate_chain = translate | model | parser
joke_chain = joke | model | parser
uranai_chain = uranai | model | parser

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
# translate
add_routes(
    app,
    translate_chain,
    path="/translate",
)
# joke
add_routes(
    app,
    joke_chain,
    path="/joke",
)
# uranai
add_routes(
    app,
    uranai_chain,
    path="/uranai",
)

# 5. Adding another route

@app.get("/hello")
def hello():
    return {"message": "Hello World!"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
