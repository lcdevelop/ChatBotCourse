# coding:utf-8

from gensim.models import word2vec
import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = word2vec.LineSentence('segment_result_lined')
#model = word2vec.Word2Vec(sentences, size=200, workers=4, iter=20)
#model.save("word_vec_model/model")
model_2 = word2vec.Word2Vec.load("word_vec_model/model")
y = model_2.most_similar(u"学习", topn=10)
for (word, score) in y:
    print word
    print score
#print model_2.syn0norm[model_2.vocab[u"小兔"].index]
