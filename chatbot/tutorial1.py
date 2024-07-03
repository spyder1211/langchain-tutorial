from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "1"}}

response = with_message_history.invoke(
    [HumanMessage("質問：私はspyder1211です。職業はプログラマーです")],
    config=config,
)

print(response.content)

response = with_message_history.invoke(
    [HumanMessage("質問：私は誰ですか？名前だけ簡潔に答えてください")],
    config=config,
)

print(response.content)

config = {"configurable": {"session_id": "2"}}

response = with_message_history.invoke(
    [HumanMessage("質問：私は誰ですか？わからなければ「わからない」と答えてください。")],
    config=config,
)

print(response.content)

config = {"configurable": {"session_id": "1"}}

response = with_message_history.invoke(
    [HumanMessage("質問：私は誰ですか？名前だけ簡潔に答えてください")],
    config=config,
)

print(response.content)