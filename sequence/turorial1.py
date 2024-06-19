from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic} in japanese")

chain = prompt | model | StrOutputParser()

output = chain.invoke({"topic": "chicken"})
print(output)
