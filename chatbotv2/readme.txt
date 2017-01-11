python ../word_segment.py zhenhuanzhuan.txt zhenhuanzhuan.segment
../word2vec/word2vec -train ./zhenhuanzhuan.segment -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15



head -10000 ../subtitle/raw_subtitles/subtitle.corpus > subtitle.corpus.10000
python ../word_segment.py subtitle.corpus.10000 subtitle.corpus.10000.segment
../word2vec/word2vec -train ./subtitle.corpus.10000.segment -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15
cat subtitle.corpus.10000.segment | awk '{if(last!="")print last"|"$0;last=$0}' | sed 's/| /|/g' > subtitle.corpus.10000.segment.pair
