from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは優秀なアシスタントです。質問に全力で答えます。"),
    MessagesPlaceholder(variable_name="messages"),
])


model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = prompt | model | StrOutputParser()

response = chain.invoke({"messages":[HumanMessage(content="私は高校1年生です。将来どのような職業に就くといいですか？")]})
print(response)

config = {"configurable": {"session_id": "1"}}

with_message_history = RunnableWithMessageHistory(chain, get_session_history)
response = with_message_history.invoke(
    [HumanMessage("もうちょっと具体的に教えてください。")],
    config=config,
)
print(response)