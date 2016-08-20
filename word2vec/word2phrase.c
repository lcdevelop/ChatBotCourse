//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 60

const int vocab_hash_size = 500000000; // Maximum 500M entries in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, min_count = 5, *vocab_hash, min_reduce = 1;
long long vocab_max_size = 10000, vocab_size = 0;
long long train_words = 0;
real threshold = 100;

unsigned long long next_random = 1;

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 1;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 10000;
    vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash]=vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, vocab_size * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  FILE *fin;
  long long a, i, start = 1;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      start = 1;
      continue;
    } else start = 0;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("Words processed: %lldK     Vocab size: %lldK  %c", train_words / 1000, vocab_size / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (start) continue;
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    strcpy(last_word, word);
    i = SearchVocab(bigram_word);
    if (i == -1) {
      a = AddWordToVocab(bigram_word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("\nVocab size (unigrams + bigrams): %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fclose(fin);
}

void TrainModel() {
  long long pa = 0, pb = 0, pab = 0, oov, i, li = -1, cn = 0;
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  real score;
  FILE *fo, *fin;
  printf("Starting training using file %s\n", train_file);
  LearnVocabFromTrainFile();
  fin = fopen(train_file, "rb");
  fo = fopen(output_file, "wb");
  word[0] = 0;
  while (1) {
    strcpy(last_word, word);
    ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      fprintf(fo, "\n");
      continue;
    }
    cn++;
    if ((debug_mode > 1) && (cn % 100000 == 0)) {
      printf("Words written: %lldK%c", cn / 1000, 13);
      fflush(stdout);
    }
    oov = 0;
    i = SearchVocab(word);
    if (i == -1) oov = 1; else pb = vocab[i].cn;
    if (li == -1) oov = 1;
    li = i;
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    i = SearchVocab(bigram_word);
    if (i == -1) oov = 1; else pab = vocab[i].cn;
    if (pa < min_count) oov = 1;
    if (pb < min_count) oov = 1;
    if (oov) score = 0; else score = (pab - min_count) / (real)pa / (real)pb * (real)train_words;
    if (score > threshold) {
      fprintf(fo, "_%s", word);
      pb = 0;
    } else fprintf(fo, " %s", word);
    pa = pb;
  }
  fclose(fo);
  fclose(fin);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD2PHRASE tool v0.1a\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-threshold <float>\n");
    printf("\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\nExamples:\n");
    printf("./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n");
    return 0;
  }
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threshold", argc, argv)) > 0) threshold = atof(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  TrainModel();
  return 0;
}
