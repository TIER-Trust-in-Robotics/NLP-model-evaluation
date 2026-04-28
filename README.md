## Goal

Train a binary gate on IEMOCAP prosodic features:

```text
neutral     -> skip transcription/NLP when confidence is high
non-neutral -> pass to transcription/NLP
```

Treat `neu` as neutral. Treat every other IEMOCAP tag as non-neutral:

```text
ang, hap, exc, sad, fru, fea, sur, dis, xxx, oth
```

The model should optimize non-neutral recall, not accuracy. It is acceptable to
process some neutral speech. It is much worse to skip emotional speech.

## Metrics

Primary metric:

```text
non_neutral_recall = fraction of non-neutral utterances passed to NLP
```

Secondary metric:

```text
neutral_skip_rate = fraction of neutral utterances skipped
```

Pick the threshold that gives the best `neutral_skip_rate` while keeping
`non_neutral_recall` at above 90%

Also report:

```text
skipped_non_neutral_count
skipped_neutral_count
overall_process_rate
confusion matrix
ROC AUC
```

## Model Variants

Start with logistic regression because it is small, fast, and easy to inspect.
Then compare against:

```text
LinearSVC + probability calibration
HistGradientBoostingClassifier
RandomForestClassifier with shallow trees
MLP
LSTM/GRU
```
## Loading data
The `loader.py` file requires that you put all the .npz (found in the google drive) in a dir called /data (or put it in the same dir if you want).
