import gensim
# model = gensim.models.Word2Vec.load("wiki.th.text.model")
model = gensim.models.Word2Vec.load("th.bin")
print(model.most_similar("แมว"))