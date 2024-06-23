import newspaper
print(newspaper.__version__)

url = "https://www.kayac.com/news/2024/06/aks"
article = newspaper.Article(url)

# ダウンロードと解析
article.download()
article.parse()

print(article.title)
print(article.text)