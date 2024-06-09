import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

messages = [
    SystemMessage(content="Translate the following from English into Japanese"),
    HumanMessage(content="hi!"),
]

chain = model | parser
output = chain.invoke(messages)
print(output)
