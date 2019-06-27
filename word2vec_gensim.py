# -*- coding: utf-8 -*-

#~~~ CMD
#~ set PYTHONIOENCODING=utf8
#~ py -3 -c "import sys;print(sys.stdout.encoding)"
#~ python test1.py > output_redirected.txt

#~~~ https://python3.wannaphong.com/2017/01/word2vec-gensim-python.html

#import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
#sys.stdout.encoding='utf-8'
#print(sys.stdout.encoding)
def simple_demo():
  from gensim.models import Word2Vec
  from pythainlp.tokenize import word_tokenize
  a = ['ฉันรักภาษาไทยเพราะฉันเป็นคนไทยและฉันเป็นคนไทย' ,'ฉันเป็นนักเรียนที่ชื่นชอบวิทยาศาสตร์และเทคโนโลยี' ,'ฉันไม่ใช่โปรแกรมเมอร์เพราะฉันทำมากกว่าคิดเขียนพัฒนาโปรแกรมทดสอบโปรแกรม','ฉันชื่นชอบวิทยาศาสตร์ชอบค้นคว้าตั้งสมมุติฐานและหาคำตอบ']
  b = [list(word_tokenize(i)) for i in a] # ทำการตัดคำแล้วเก็บใน list จะได้เป็น [['ฉัน',...],['ฉัน',...]...]
  model = Word2Vec(b, min_count=1)
  aa=model.similar_by_word('เป็น')
  print(aa)

def process_wiki():
  import logging
  import os.path
  import sys
  
  from gensim.corpora import WikiCorpus
  print(__name__)
  print("running %s" % ' '.join(sys.argv))

  # if __name__ == '__main__':
  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)

  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logger.info("running %s" % ' '.join(sys.argv))

  # check and process input arguments
  if len(sys.argv) < 3:
      print((globals()['__doc__'] % locals()))
      sys.exit(1)
  inp, outp = sys.argv[1:3]
  space = b' '
  i = 0

  output = open(outp, 'w', encoding='utf-8')
  wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
  for text in wiki.get_texts():
      list1=space.join(text)
      output.write((list1.decode('utf-8')) + "\n")
      i+=1
      if (i % 10000 == 0):
          logger.info("Saved " + str(i) + " articles")

  output.close()
  logger.info("Finished Saved " + str(i) + " articles")