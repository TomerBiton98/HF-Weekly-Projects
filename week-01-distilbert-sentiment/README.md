Week 01 — DistilBERT Sentiment Analyzer
Hugging Face Weekly Model Series – by Tomer Biton

This project demonstrates a simple but powerful sentiment analysis pipeline using:

🧠 DistilBERT - a distilled, lightweight version of BERT

*Fast inference (60% faster than BERT)
* High accuracy (97% of BERT’s performance)
*CSV comment analysis support
*Command-line interface (CLI)

Project Overview
This week’s focus is using:
distilbert-base-uncased-finetuned-sst-2-english

A pre-trained binary sentiment classifier from Hugging Face, fine-tuned on the SST-2 dataset.
You can analyze either:
A single text message, or
A CSV file containing comments

The project outputs:
Sentiment label (POSITIVE / NEGATIVE)
Confidence score
