import argparse
import pandas as pd
from transformers import pipeline


analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="af0f99b"
)

def analyze_text(text):
    r = analyzer(text)[0]
    return r["label"], round(r["score"], 2)

def analyze_csv(path):
    df = pd.read_csv(path)
    df["label"], df["score"] = zip(*df["comment"].apply(analyze_text))
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text")
    p.add_argument("--csv")
    a = p.parse_args()

    if a.text:
        label, score = analyze_text(a.text)
        print(f"{a.text}\n→ {label} ({score})")

    elif a.csv:
        df = analyze_csv(a.csv)
        print(df.to_string(index=False))

    else:
        print("Usage:\n  python app.py --text \"Hi\"\n  python app.py --csv file.csv")

if __name__ == "__main__":
    main()
