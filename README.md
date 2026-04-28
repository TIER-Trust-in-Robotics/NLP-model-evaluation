#### Metrics

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
