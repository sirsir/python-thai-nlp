# -*- coding: utf-8 -*-

#~~~ CMD
#~ set PYTHONIOENCODING=utf8
#~ py -3 -c "import sys;print(sys.stdout.encoding)"
#~ python test1.py > output_redirected.txt


import word2vec_gensim

# word2vec_gensim.py.simple_demo()

word2vec_gensim.process_wiki()
#!! python test1.py "data/thwiki-latest-pages-articles.xml.bz2" "output/wiki.th.text"
# word2vec_gensim.process_wiki("data/thwiki-latest-pages-articles.xml.bz2","output/wiki.th.text")