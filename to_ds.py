import csv
import re
from pathlib import Path
from datasets import Dataset, Audio
import ast


DATA_ROOT = Path("/path/to/datasets")  # path to datasets folder
IEMOCAP_ROOT = DATA_ROOT / "IEMOCAP_full_release"
MSP_ROOT = DATA_ROOT / "MSP-PODCAST"


# -------------------------
# IEMOCAP
# -------------------------

# renaming t
IEMOCAP_LABELS = [
    "ang",  # anger
    "hap",  # happiness
    "exc",  # excitement
    "neu",  # neutral
    "sad",  # sadness
    "fru",  # frustration
    "fea",  # fear
    "sur",  # surprise
    "dis",  # disgust
    "xxx",  # other/undefined
    "oth",  # other
]

IEMOCAP_LABEL_NAMES = [
    "anger",
    "happiness",
    "excitement",
    "neutral",
    "sadness",
    "frustration",
    "fear",
    "surprise",
    "disgust",
    "undefined",
    "other",
]

IEMOCAP_LABEL2IDX = {l: i for i, l in enumerate(IEMOCAP_LABELS)}


def _parse_iemocap_emo_file(emo_path: Path) -> dict:
    """Parse an IEMOCAP EmoEvaluation file. Returns {utterance_id: [emotion, V, A, D]}."""
    utt2emo = {}
    with open(emo_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue
            # Format: [start - end]\t<UTTERANCE_ID>\t<EMOTION>\t[V(alence), A(ctivation), D(ominance)]
            # \t = tab
            parts = line.split("\t")
            if len(parts) >= 3:
                utt_id = parts[1].strip()
                emo = parts[2].strip()
                VAD = ast.literal_eval(parts[3])
                utt2emo[utt_id] = [emo] + VAD
    return utt2emo


def _parse_iemocap_transcript(trans_path: Path) -> dict[str, str]:
    """Parse an IEMOCAP transcription file. Returns {utterance_id: text}."""
    utt2text = {}
    with open(trans_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: UTT_ID [start-end]: text
            match = re.match(
                r"^(\S+)\s+\[[\d.\-]+\]:\s*(.+)$", line
            )  # it works i guess, thanks claude Opus 4.6 wt Extended Thinking™    :^).
            # The regex grabs two things, the first non-white space (UTT_ID), ignores anything within brackets (i.e. the time stamp), as well as a colon and white space before the next block of text that is grabbed until the end of line symbol ($)
            if match:
                utt_id = match.group(1)
                text = match.group(2).strip()
                utt2text[utt_id] = text
    return utt2text


def load_iemocap() -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load IEMOCAP dataset from local files.

    Sessions 1-4 are used for training, Session 5 for testing.

    Returns: (train_ds, test_ds, num_labels, label_names)
    """
    train_texts, train_labels, train_audio = [], [], []
    test_texts, test_labels, test_audio = [], [], []

    for session_num in range(1, 6):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        wav_dir = session_dir / "sentences" / "wav"

        # Skip Attribute / Categorical / Self-evaluation subdirs
        emo_files = [
            f
            for f in emo_dir.glob("*.txt")
            if f.stem not in ("Attribute", "Categorical", "Self-evaluation")
        ]

        for emo_file in sorted(emo_files):
            dialog_name = emo_file.stem
            trans_file = trans_dir / f"{dialog_name}.txt"
            if not trans_file.exists():
                continue

            utt2emo = _parse_iemocap_emo_file(emo_file)
            utt2text = _parse_iemocap_transcript(trans_file)

            for utt_id, emo in utt2emo.items():
                if utt_id not in utt2text:
                    continue
                if emo not in IEMOCAP_LABEL2IDX:
                    continue

                audio_path = wav_dir / dialog_name / f"{utt_id}.wav"
                if not audio_path.exists():
                    continue

                text = utt2text[utt_id]
                label = IEMOCAP_LABEL2IDX[emo]

                if session_num <= 4:
                    train_texts.append(text)
                    train_labels.append(label)
                    train_audio.append(str(audio_path))
                else:
                    test_texts.append(text)
                    test_labels.append(label)
                    test_audio.append(str(audio_path))

    train_ds = Dataset.from_dict(
        {"text": train_texts, "label": train_labels, "audio": train_audio}
    ).cast_column("audio", Audio())
    test_ds = Dataset.from_dict(
        {"text": test_texts, "label": test_labels, "audio": test_audio}
    ).cast_column("audio", Audio())

    return train_ds, test_ds, len(IEMOCAP_LABELS), IEMOCAP_LABEL_NAMES


# -------------------------
# MSP-PODCAST
# -------------------------

MSP_LABELS = [
    "A",  # Anger
    "S",  # Sadness
    "H",  # Happiness
    "U",  # Surprise
    "F",  # Fear
    "D",  # Disgust
    "C",  # Contempt
    "N",  # Neutral
]

MSP_LABEL_NAMES = [
    "anger",
    "sadness",
    "happiness",
    "surprise",
    "fear",
    "disgust",
    "contempt",
    "neutral",
]

MSP_LABEL2IDX = {l: i for i, l in enumerate(MSP_LABELS)}


# FIX: Add the secondary emotion labels as well
def load_msp_podcast() -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load MSP-PODCAST dataset from local files.

    Uses Train split for training and Test1 for evaluation.
    Excludes 'X' (no agreement) and 'O' (other) labels from the labels

    Returns: (train_ds, test_ds, num_labels, label_names)
    """
    labels_path = MSP_ROOT / "Labels" / "labels_consensus.csv"
    transcripts_dir = MSP_ROOT / "Transcripts"
    audios_dir = MSP_ROOT / "Audios"

    train_texts, train_labels, train_audio = [], [], []
    test_texts, test_labels, test_audio = [], [], []

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emo = row["EmoClass"]
            split = row["Split_Set"]

            # Skip excluded labels
            if emo not in MSP_LABEL2IDX:
                continue

            # Only use Train and Test1 (Test 2 could work, but maybe later)
            if split not in ("Train", "Test1"):
                continue

            filename = row["FileName"]
            transcript_file = transcripts_dir / filename.replace(".wav", ".txt")
            audio_file = audios_dir / filename
            if not transcript_file.exists() or not audio_file.exists():
                continue

            text = transcript_file.read_text().strip()
            if not text:
                continue

            label = MSP_LABEL2IDX[emo]

            if split == "Train":
                train_texts.append(text)
                train_labels.append(label)
                train_audio.append(str(audio_file))
            else:
                test_texts.append(text)
                test_labels.append(label)
                test_audio.append(str(audio_file))

    train_ds = Dataset.from_dict(
        {"text": train_texts, "label": train_labels, "audio": train_audio}
    ).cast_column("audio", Audio())
    test_ds = Dataset.from_dict(
        {"text": test_texts, "label": test_labels, "audio": test_audio}
    ).cast_column("audio", Audio())

    return train_ds, test_ds, len(MSP_LABELS), MSP_LABEL_NAMES
