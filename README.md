# NLP Fundamentals — Practical Notebooks

A series of hands-on Jupyter notebooks working through the foundational techniques of Natural Language Processing in Python. Each notebook is self-contained, runs end-to-end, and focuses on building intuition for one class of technique through real text — primarily a speech by Dr. A. P. J. Abdul Kalam and short custom corpora.

The notebooks progress from character-level text preprocessing up to distributed word representations, mirroring the traditional NLP pipeline: *raw text → tokens → normalized tokens → syntactic tags → entities → vectors.*

---

## Contents

### 1. Tokenization (`01_tokenization.ipynb`)

Tokenization is the first step of almost every NLP pipeline. This notebook compares four NLTK tokenizers on the same corpus to show that "splitting text" is not a single, well-defined operation.

- `sent_tokenize` — paragraph to sentences
- `word_tokenize` — sentence to words, handles contractions and punctuation
- `wordpunct_tokenize` — aggressive punctuation-splitting variant
- `TreebankWordTokenizer` — Penn Treebank conventions (e.g., keeps trailing punctuation attached)

**Takeaway:** different tokenizers produce different token counts and segmentations. Choice of tokenizer depends on the downstream task.

### 2. Stemming, Lemmatization, and Stopword Removal (`02_stemming_lemmatization.ipynb`)

Compares three stemming algorithms and a dictionary-based lemmatizer on the same word list, then applies the pipeline to Kalam's speech.

- **PorterStemmer** — classic, aggressive, produces non-words (`histori`, `congratul`)
- **RegexpStemmer** — custom suffix-stripping rules
- **SnowballStemmer** — improved Porter, handles edge cases better (`fair` vs. Porter's `fairli`)
- **WordNetLemmatizer** — dictionary-based, requires POS tag, produces real words (`go`, `write`)

**Pipeline demonstrated:** sentence tokenization → stopword filtering → stemming/lemmatization → reconstruction. Output is compared side-by-side across all three stemmers and the lemmatizer.

**Takeaway:** stemming is fast and crude; lemmatization is slower but semantically correct. POS-tag-aware lemmatization (`pos='v'`) produces noticeably better results than default noun-based lemmatization.

### 3. POS Tagging and Named Entity Recognition (`03_pos_ner.ipynb`)

Applies NLTK's averaged perceptron POS tagger and `ne_chunk` NER to Kalam's speech.

- Part-of-speech tagging with the Penn Treebank tagset (NNP, VBD, JJ, etc.)
- Named Entity Recognition extracting `PERSON`, `ORGANIZATION`, `GPE`, `DATE` entities
- Visualized as a parse tree with `nltk.ne_chunk(...).draw()`

**Takeaway:** NLTK's NER is a useful baseline but makes notable errors on informal or proper-noun-heavy text. Works well on well-formed sentences like "The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel".

### 4. Word Embeddings with Word2Vec (`04_word_embeddings.ipynb`)

Loads Google's pretrained Word2Vec model (300-dimensional, trained on 100B tokens of Google News) via Gensim and explores the geometry of the embedding space.

- `api.load('word2vec-google-news-300')` — pretrained model, ~1.6 GB
- Vector lookup for any in-vocabulary word
- **Semantic similarity**: `wv.similarity("hockey", "sports")` → 0.535
- **Nearest neighbors**: `wv.most_similar('cricket')` returns cricketing, cricketers, Test_cricket, etc.
- **Vector arithmetic**: `king - man + woman` ≈ `queen` (cosine ≈ 0.73)

**Takeaway:** dense distributed representations capture both topical similarity and linear analogical structure. This is the bridge between classical NLP and modern transformer-based models.

---

## Setup

### Requirements

```
python >= 3.10
nltk
gensim
numpy
scipy
```

---

## What I Learned

These notebooks were built as a hands-on way to internalize classical NLP concepts rather than read about them. Key insights after working through all four:

- **Preprocessing choices cascade.** A different tokenizer changes what's left after stopword removal, which changes what the stemmer produces, which changes every downstream representation. There is no "neutral" preprocessing.

- **Stemming vs. lemmatization is a speed/quality tradeoff.** Stemming is essentially regex; lemmatization is a dictionary lookup with POS context. For search and retrieval, stemming is often enough. For generation or semantic analysis, lemmatization is worth the cost.

- **Rule-based NER degrades quickly outside newswire-style text.** NLTK's NER works well on the Eiffel Tower example but makes obvious errors on the Kalam speech (e.g., missing "Dr. Vikram Sarabhai" as a person). This is why modern systems use contextual models.

- **Word2Vec is still useful pedagogically.** The `king - man + woman ≈ queen` example is genuinely striking when you compute it yourself. It makes the geometric interpretation of embeddings concrete in a way transformer attention maps do not.

---

Sample text is drawn from a public-domain speech by Dr. A. P. J. Abdul Kalam.
