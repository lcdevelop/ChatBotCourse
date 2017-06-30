1. 安装anaconda3
2. 通过pip安装tensorflow等必要的包
3. 利用Word2vec通过大量文本训练词向量放大data/word2vec中,搞不定找我这里有用几个G的文章训练出来的效果较好的向量文件
4. 执行python classify.py train训练模型文件data/model
5. 执行python classify.py加载模型并提供分类服务，访问http://127.0.0.1:5001?q=高效能人士的七个习惯
