from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

search = TavilySearchResults(max_results=2)

tools = [search]

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="こんにちわ！わたしはそういちろうです")]}, config
):
    print(chunk)
    print("-----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="わたしの名前はなんですか？")]}, config
):
    print(chunk)
    print("-----")