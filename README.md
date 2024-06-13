# langchain-tutorial

## 参考
https://python.langchain.com/v0.2/docs/tutorials/llm_chain/

## turorial
https://python.langchain.com/v0.2/docs/tutorials/

## langserver起動方法

```sh
$ python LangServe/serve.py
```

## 動作確認方法

```sh
curl -X 'POST' \
  'http://localhost:8000/chain/invoke' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": {
    "language": "日本語",
    "text": "Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, using these LLMs in isolation is often insufficient for creating a truly powerful app - the real power comes when you can combine them with other sources of computation or knowledge"
  },
  "config": {},
  "kwargs": {}
}'
```

## API仕様
http://localhost:8000/docs

## プレイグラウンド
http://localhost:8000/chain/playground