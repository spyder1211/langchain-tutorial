from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
output = remote_chain.invoke({"language": "日本語", "text": "im feeling good today!"})
print(output)
