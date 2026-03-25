# Sentiment Analysis of Singapore Airlines Passenger Reviews

### Research Question
**How accurately can NLP models classify passenger sentiment from Singapore Airlines reviews, and what thematic drivers underpin negative customer experiences?**

---

## Project Overview

This project applies Natural Language Processing (NLP) techniques to analyse 10,000 Singapore Airlines passenger reviews. Two sentiment classification approaches are compared — VADER (lexicon-based) and RoBERTa (transformer-based transfer learning) — and BERTopic is used for unsupervised theme discovery. The cross-analysis of sentiment and topics reveals that operational issues (lost luggage, booking disputes) drive the most negative sentiment, while in-flight service is overwhelmingly positive.

### Key Results

| Metric             | VADER  | RoBERTa |
|--------------------|--------|---------|
| Accuracy           | 0.8153 | 0.7999  |
| Weighted F1-Score  | 0.7786 | 0.8005  |

**Final model:** RoBERTa (higher F1-score, lower false-positive rate for negative reviews)

**BERTopic:** 19 topics discovered after parameter tuning (Coherence: 0.4829, Diversity: 0.4895)

## Repository Structure

```
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
├── notebooks/
│   └── SIA_Sentiment_Analysis.ipynb           # Main analysis notebook
├── data/
│   └── (download instructions below)
├── figures/                                   # Generated visualisations
│   ├── rating_distribution.png
│   ├── review_length_distribution.png
│   ├── top_words.png
│   ├── wordcloud.png
│   ├── sentiment_distribution.png
│   ├── vader_confusion_matrix.png
│   ├── roberta_confusion_matrix.png
│   ├── model_comparison.png
│   ├── topic_sentiment_heatmap.png
│   ├── topic_barchart.html
│   ├── topic_hierarchy.html
│   └── topic_heatmap.html
└── results/
    ├── processed_reviews_with_sentiment.csv
    └── model_comparison.csv
```

## How to Reproduce

### Step 1: Clone the repository
```bash
git clone https://github.com/Krit-Khunkitti/Singapore-Airline-Review-Analysis
cd Singapore-Airline-Review-Analysis
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the dataset
1. Visit: https://www.kaggle.com/datasets/kanchana1990/singapore-airlines-reviews
2. Download `singapore_airlines_reviews.csv`
3. Place the file in the `data/` folder

### Step 4: Run the notebook
```bash
jupyter notebook notebooks/SIA_Sentiment_Analysis.ipynb
```
Run all cells sequentially (**Kernel → Restart & Run All**).

**Note:** The RoBERTa analysis may take 10–20 minutes on CPU. For faster execution, use Google Colab with a T4 GPU runtime.

## Methods

| Method | Type | Purpose | Reference |
|--------|------|---------|-----------|
| VADER | Lexicon-based (baseline) | Fast, rule-based sentiment classification | Hutto & Gilbert (2014) |
| RoBERTa | Transformer (transfer learning) | Context-aware sentiment classification | Liu et al. (2019) |
| BERTopic | Unsupervised neural topic model | Discover thematic clusters in reviews | Grootendorst (2022) |

## Key Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- vaderSentiment
- transformers (Hugging Face), torch
- bertopic, sentence-transformers
- umap-learn, hdbscan
- scikit-learn, gensim

See `requirements.txt` for full list with version pinning.

## Data Source

Kanchana. (2024). *Singapore Airlines Reviews* [Data set]. Kaggle. https://www.kaggle.com/datasets/kanchana1990/singapore-airlines-reviews
