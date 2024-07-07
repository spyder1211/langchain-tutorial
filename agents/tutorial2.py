from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

search = TavilySearchResults(max_results=2)

tools = [search]

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content="40歳女性が欲しがりそうなプレゼントは何ですか？")], "context": [{"source": "https://example.com/"}]})

print(response["messages"])