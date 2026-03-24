# Learning Resources

## NLP & LLMs
- [3blue1brown Neural Network Series](https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5), namely pt. 5 & 6.
- [Andrej Karpathy's Zero to Hero Series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), very techical (very much optional).
- [Jay Alammar's BERT blog post](https://jalammar.github.io/illustrated-bert/), requires understanding transformer.
- [Hugging Face Learning Center](huggingface.co/learn), namely the LLM section.

# NLP models to be tested

## Datasets to use
- [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) (pending) 
- [DailyDialog](https://huggingface.co/datasets/frankdarkluo/DailyDialog)
- [GoEmotion](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [Stanford Sentiment Treebank-2 (SST-2)](https://huggingface.co/datasets/stanfordnlp/sst2)

### Dataset evaluation
- [distilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)
- [tinyBERT](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2)
- [bert-mini](https://huggingface.co/prajjwal1/bert-mini)
- [MiniLM-L12](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)
- [MiniLM-L6](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)


#### Metrics
Weighed F1, macro f1, inference latency, macro recall, macro precision, model-size, and per-class performance. These are to be tested on CPU and CUDA (Jetson Orin nano Super 8GB).
