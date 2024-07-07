from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

search = TavilySearchResults(max_results=2)

tools = [search]

model = ChatBedrock(
    region_name='us-east-1',
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
)

message = HumanMessage(content="今日の東京の天気は？")

model_with_tools = model.bind_tools(tools)
output = model_with_tools.invoke([message])
print(f"ContentString: {output.content}")
print(f"ToolCalls: {output.tool_calls}")