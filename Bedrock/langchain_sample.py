from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

chat = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

template = ChatPromptTemplate.from_messages([
    ('system', 'Translate the following into {language}:'),
    ('user', '{text}')
])

response = chat.invoke(template.invoke({"language": "英語", "text": "サンプルの中では、独自の関数が利用されているため、シンプルに書き替えました。"}))

print(response.content)
