import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

messages = [
    SystemMessage(content="Translate the following from English into Japnaese"),
    HumanMessage(content="hi!"),
]

# result = model.invoke(messages)
# output = parser.invoke(result)
# print(output)

chain = model | parser
output = chain.invoke(messages)
print(output)
