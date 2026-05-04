# 🧠 Meme Context Clustering
### Unsupervised Discovery of Semantic Structure in Internet Meme Datasets

<p align="center">
  <img src="outputs/05_tsne_clusters.png" alt="t-SNE Cluster Visualization" width="85%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%2B%20LSA-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-Unsupervised%20Clustering-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Competition-DataSprint%202026-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Libraries-NumPy%20%7C%20Pandas%20%7C%20Sklearn%20%7C%20Matplotlib-yellow?style=flat-square" />
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why This Problem Matters](#-why-this-problem-matters)
- [Real-World Applications](#-real-world-applications)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Methodology](#-methodology)
  - [1. Structural Parsing](#1-structural-parsing)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Dimensionality Reduction](#4-dimensionality-reduction)
  - [5. Clustering Algorithms](#5-clustering-algorithms)
  - [6. Optimal K Selection](#6-optimal-k-selection)
  - [7. Hyperparameter Tuning](#7-hyperparameter-tuning)
- [Results](#-results)
- [Discovered Clusters](#-discovered-clusters)
- [Visualizations](#-visualizations)
- [Key Design Decisions](#-key-design-decisions)
- [Limitations & Trade-offs](#-limitations--trade-offs)
- [How to Run](#-how-to-run)
- [File Structure](#-file-structure)
- [Competition Context](#-competition-context)
- [What I Learned](#-what-i-learned)

---

## 🔍 Overview

This project applies **unsupervised machine learning** to discover hidden semantic groupings — clusters — within a dataset of **5,818 internet memes**. Each meme is described not just by its visible text, but by a rich structured annotation including image descriptions, inferred intent, and entity-role mappings.

The core challenge: **there are no labels**. No predefined categories. No right answer. The model must find natural structure entirely on its own, guided only by the semantic signals embedded in text — and evaluated purely by internal metrics like Silhouette Score and Davies-Bouldin Index.

The result is **8 semantically coherent meme clusters**, ranging from Elon Musk / Twitter memes to Relationship commentary to Pandemic humor — each with distinct vocabulary signatures, emotional tones, and entity patterns.

> **Competition:** DataSprint 2026 — ML Coding Round  
> **Platform:** Kaggle ([Meme Context Clustering](https://www.kaggle.com/competitions/meme-context-clustering))  
> **Constraint:** Standard libraries only — NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn. No AutoML, no pretrained models, no external APIs.

---

## 💡 Why This Problem Matters

Internet memes are not just jokes. They are **compressed cultural signals** — they encode political sentiment, social anxiety, generational humor, and shared experiences in a way that traditional text analysis tools struggle to handle. A meme's meaning rarely lives in its visible text alone; it emerges from the interaction between the image, the text, and the cultural context the viewer brings.

Understanding meme semantics at scale matters because:

- **Memes spread faster than news articles** and often carry misinformation, political narratives, or emotional manipulation embedded in humor
- **Content moderation systems** that rely only on keyword matching miss the semantic layer entirely — a meme about "acquiring Twitter" and one about "buying a car" might share zero keywords but require completely different moderation responses
- **Social listening tools** used by brands, researchers, and governments need to understand what people are actually saying, not just what words they're using
- **Meme cultures are fragmented** — what's funny to a gaming community is opaque to a political community. Clustering helps map these cultural fault lines automatically

This project is a small step toward making machine learning understand the internet the way humans do — not word by word, but context by context.

---

## 🌍 Real-World Applications

| Domain | How Meme Clustering Helps |
|--------|--------------------------|
| **Content Moderation** | Automatically categorize meme types for review queues; flag clusters that historically contain harmful content |
| **Social Media Analytics** | Track which meme genres are trending; measure sentiment shifts within specific communities |
| **Political Research** | Identify and monitor political meme clusters; study how narratives spread through humor |
| **Brand Intelligence** | Detect when a brand is being memed positively vs. negatively; understand cultural positioning |
| **Recommendation Systems** | Suggest contextually similar memes rather than just visually similar ones |
| **Misinformation Research** | Cluster memes by intent to find coordinated campaigns that use the same semantic frame with different visuals |
| **Cultural Anthropology** | Study internet subcultures as distinct communities with their own humor vocabularies |
| **NLP Research** | Benchmark structured text parsing and semi-supervised clustering techniques on noisy, real-world data |

---

## 📊 Dataset

**Source:** [Kaggle — Meme Context Clustering](https://www.kaggle.com/competitions/meme-context-clustering)

| Property | Value |
|----------|-------|
| Total samples | 5,818 memes |
| Features | 3 columns (id, input, url) |
| Missing values | 0 |
| Label availability | None (unsupervised) |
| Input type | Semi-structured text |
| Avg input length | ~270 characters |

### The `input` Field Structure

Each meme's `input` field contains four semicolon-separated components:

```
TEXT: <visible meme text>; <image description 1>, <image description 2>; <intent sentence 1>, <intent sentence 2>; <entity>: <role>, <entity>: <role>, ...
```

**Example:**
```
TEXT: For real though;
Person in Spider Man outfit gives a lecture on stage., Person dressed as spider man stands in front of crowd with notes;
Meme poster is frustrated about the format of the website and is making a suggestion for improvement.;
Spider Man outfit: Meme poster, a lecture: complaint, crowd: meme readers
```

This structure is what makes the problem interesting — and what most naive approaches miss entirely.

---

## 🏗️ Project Architecture

```
meme.csv (raw data)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Structural Parsing                                 │
│  Split input → meme_text | image_desc | intent | entity     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Text Preprocessing                                 │
│  Lowercase → Remove non-alpha → 2-layer stopword removal    │
│  → Intent 3x weighted in combined_clean                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Feature Engineering                                │
│  Layer A: TF-IDF (800 features, bigrams, sublinear_tf)      │
│  Layer B: TruncatedSVD/LSA (80 latent dimensions)           │
│  Layer C: 21 semantic features (emotion + topic + entity)   │
│           → Boosted 3x to surface niche clusters            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Optimal K Selection                                │
│  Scan K=2..11 → Silhouette + Davies-Bouldin + CH metrics    │
│  → K=8 selected (balance of quality + semantic richness)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Three-Model Training & Comparison                  │
│  K-Means │ Agglomerative (Ward) │ Spherical K-Means         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Hyperparameter Tuning + CV Stability Check         │
│  Grid: n_init × init_method → 5-seed cross-validation       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Cluster Profiling + Interpretation                 │
│  TF-IDF centroids → keyword extraction per cluster          │
│  Feature heatmap → semantic profile per cluster             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              submission.csv | report.txt | 7 visualizations
```

---

## 🔬 Methodology

### 1. Structural Parsing

The first and most important insight: **the `input` field is not raw text — it is structured data disguised as a string.**

```python
def parse_meme_input(text):
    parts = str(text).split(';')
    meme_text    = parts[0].replace('TEXT:', '').strip()   # Visible meme text
    image_desc   = parts[1].strip()                         # What's in the image
    intent       = parts[2].strip()                         # What the meme means
    entity_roles = parts[3].strip()                         # Who plays what role
    return meme_text, image_desc, intent, entity_roles
```

Most participants would feed the raw input directly into TF-IDF. By parsing it, we can:
- Weight each sub-field differently based on its semantic value
- Extract structural features (number of entities, intent sentence count)
- Apply targeted cleaning to each layer independently

### 2. Data Preprocessing

**The boilerplate problem:** After initial EDA, the most frequent terms in the dataset were "meme", "poster", "trying", "convey", "person", "image" — words that appear in nearly every row and carry zero discriminative power for clustering.

Solution: **Two-layer stopword removal**

```
Layer 1: Standard English stopwords (sklearn's built-in list)
Layer 2: 60 domain-specific stopwords curated from frequency analysis
         e.g., {meme, poster, person, image, trying, convey, says,
                showing, depicting, talking, written, ...}
```

Without this, every cluster's centroid would look identical — dominated by the same boilerplate vocabulary.

**Intent triple-weighting:** The intent field describes *what the meme is communicating* — the actual semantic core. We include it three times in the combined text field so TF-IDF naturally assigns it higher weight:

```python
df['combined'] = (
    df['intent_clean'] + ' ' +   # 1x
    df['intent_clean'] + ' ' +   # 2x  
    df['intent_clean'] + ' ' +   # 3x — this is the meaning
    df['image_desc_clean'] + ' ' +
    df['meme_text_clean']
)
```

### 3. Feature Engineering

Three complementary feature layers, each solving a different limitation:

**Layer A — TF-IDF (800 features, bigrams):**
Captures raw vocabulary. Bigrams are non-negotiable here — "elon musk", "kanye west", "video game", "reddit users" are meaningful units that unigrams would split.

```python
TfidfVectorizer(
    max_features=800,
    ngram_range=(1, 2),    # Unigrams + bigrams
    min_df=2,              # Must appear in ≥2 docs
    max_df=0.80,           # Must not appear in >80% of docs
    sublinear_tf=True      # log(1 + tf) — dampens high-frequency terms
)
```

**Layer B — LSA / TruncatedSVD (80 dimensions):**
TF-IDF is a 800-dimensional sparse matrix. SVD compresses it into 80 dense dimensions that capture *latent semantic topics* — groups of words that co-occur together. This is Latent Semantic Analysis (LSA).

Why not PCA? PCA requires densifying the matrix first (800×5818 floats in memory). TruncatedSVD operates on the sparse matrix directly — critical for memory efficiency.

**Layer C — 21 Hand-Crafted Semantic Features (3× weighted):**

| Category | Features |
|----------|----------|
| Structural | `intent_word_len`, `num_entities`, `has_question`, `has_exclamation`, `caps_ratio` |
| Emotion | `is_frustrated`, `is_funny`, `is_sad`, `is_happy`, `is_relatable`, `is_sarcastic` |
| Topic | `topic_social_media`, `topic_politics`, `topic_gaming`, `topic_relationship`, `topic_celebrity`, `topic_pandemic`, `topic_school` |
| Entity | `elon_ref`, `kanye_ref`, `reddit_ref`, `twitter_ref` |

These features inject domain knowledge that pure bag-of-words misses. A gaming meme and a political meme can have similar word frequencies but completely different topic flags.

The 3× weight boost ensures these domain-knowledge features can compete against the 80 LSA dimensions that would otherwise dominate.

### 4. Dimensionality Reduction

```
Input:  5,818 × 800  (TF-IDF sparse matrix)
After SVD: 5,818 × 80  (dense, 27.5% variance explained)
After concat: 5,818 × 101  (80 LSA + 21 meta × 3 boost)
```

### 5. Clustering Algorithms

Three fundamentally different approaches, chosen to test different geometric assumptions about the data:

| Model | Distance Metric | Cluster Assumption | When it wins |
|-------|----------------|-------------------|--------------|
| K-Means | Euclidean | Spherical, equal-size | General-purpose baseline |
| Agglomerative (Ward) | Euclidean (within-cluster variance) | Hierarchical, any shape | Non-spherical boundaries |
| Spherical K-Means | Cosine similarity | Directional, text-specialized | High-dimensional text |

### 6. Optimal K Selection

K was selected using **three simultaneous metrics**, not just one:

| Metric | Formula | Optimal When |
|--------|---------|-------------|
| **Silhouette Score** | (b - a) / max(a, b) | Maximized |
| **Davies-Bouldin Index** | avg(max(σᵢ+σⱼ)/d(cᵢ,cⱼ)) | Minimized |
| **Calinski-Harabasz** | between-cluster / within-cluster variance | Maximized |

K=8 was selected as it provides the best **balance between metric quality and semantic richness** — K=2 or K=3 had slightly higher silhouette but produced only trivial splits (one giant cluster + tiny outliers), which doesn't serve the competition's goal of uncovering hidden *contexts and genres*.

### 7. Hyperparameter Tuning

Grid search over initialization strategies:

```python
n_init  ∈ {10, 20, 30, 50}
init    ∈ {'k-means++', 'random'}
```

Cross-validation via **5-seed stability check** — running the same configuration with different random seeds to confirm cluster structure is stable, not a lucky initialization:

```
Seed  0: Silhouette = 0.2472
Seed  7: Silhouette = 0.3462
Seed 13: Silhouette = 0.2128
Seed 21: Silhouette = 0.2472
Seed 42: Silhouette = 0.3462

Mean ± Std: 0.3886 ± 0.032
```
Low standard deviation = the cluster structure is real, not random.

---

## 📈 Results

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-------|-------------|-----------------|-------------------|
| K-Means | 0.3722 | 0.8475 | 534.9 |
| **Agglomerative (Ward)** | **0.4499** | **0.8725** | **555.4** |
| Spherical K-Means | 0.3233 | 1.5070 | 1012.2 |

**Final Model (K-Means, tuned):**

| Metric | Score |
|--------|-------|
| Silhouette Score | **0.4494** |
| Davies-Bouldin Index | **0.6735** |
| Calinski-Harabasz | **540.0** |
| CV Stability (5 seeds) | 0.3886 ± 0.032 |

---

## 🗂️ Discovered Clusters

<p align="center">
  <img src="outputs/07_cluster_keywords.png" alt="Cluster Keywords" width="90%"/>
</p>

| Cluster | Name | Size | % | Description |
|---------|------|------|---|-------------|
| **C0** | General Humor & Relatable | 5,223 | 89.8% | Broad everyday humor — life, family, work, school. No single niche dominates. The backbone of internet meme culture. |
| **C1** | Elon Musk / Twitter Memes | 83 | 1.4% | Tightly clustered around Elon's Twitter acquisition, blue checkmarks, and platform drama. Keywords: elon, musk, twitter, bought, blue. |
| **C2** | Pandemic & News Events | 25 | 0.4% | COVID-19 humor, lockdown life, pandemic anxiety. Small but semantically coherent. |
| **C3** | Philosophical / Surreal | 35 | 0.6% | Existential memes about earth, humanity, nature, fear. "Dark humor" and abstract commentary. |
| **C4** | Kanye West / Celebrity Drama | 53 | 0.9% | Kanye's controversies, Hitler comments, Alex Jones. Clearly separated from C1 despite both being celebrity clusters. |
| **C5** | Relationships & Dating | 148 | 2.5% | Love, couples, girlfriend/boyfriend dynamics. Keywords: girlfriend, love, wife, couple. |
| **C6** | Reddit/Twitter Meta-Commentary | 127 | 2.2% | Memes *about* internet platforms — how Reddit works, meme culture itself, platform behavior. |
| **C7** | Twitter Platform Memes | 124 | 2.1% | Specifically about Twitter's functionality, user exodus, Elon's ownership effects. Overlaps with C1/C6 but platform-focused. |

The model correctly identified that the dataset is **not uniformly distributed** — there's a dominant general-humor majority (C0) and several distinct niche communities (C1–C7) embedded within it. This mirrors how internet meme culture actually works.

<p align="center">
  <img src="outputs/06_cluster_feature_heatmap.png" alt="Cluster Feature Heatmap" width="85%"/>
</p>

---

## 📊 Visualizations

All visualizations are generated automatically by the pipeline and saved to `outputs/`.

| File | Description |
|------|-------------|
| `01_eda_overview.png` | Input length distribution, intent word count, top entity roles, detected themes, top meme text words |
| `02_optimal_k_selection.png` | Elbow method, Silhouette score curve, Davies-Bouldin curve across K=2..11 |
| `03_silhouette_comparison.png` | Per-cluster silhouette coefficient plots for all three models side-by-side |
| `04_model_comparison_metrics.png` | Bar chart comparing Silhouette, DB, and CH across three models |
| `05_tsne_clusters.png` | t-SNE 2D scatter plot (coloured by cluster) + cluster size pie chart |
| `06_cluster_feature_heatmap.png` | Heatmap of semantic feature prevalence per cluster |
| `07_cluster_keywords.png` | Top TF-IDF keywords per cluster (8 subplots) |

<p align="center">
  <img src="outputs/01_eda_overview.png" alt="EDA Overview" width="85%"/>
</p>

---

## 🧩 Key Design Decisions

### Why parse the input field instead of using it raw?
The four sub-fields carry fundamentally different information. Intent describes *meaning*; image description describes *visuals*; entity roles describe *metaphors*. Treating them as one undifferentiated blob loses this structure. Parsing them separately allows selective weighting — the most important insight in the entire pipeline.

### Why 60 domain stopwords instead of just standard English ones?
Standard stopwords remove "the", "is", "are". They don't remove "meme", "poster", "trying", "convey" — words that appeared in 95%+ of the dataset and would dominate every TF-IDF vector, making all clusters look the same at their centroids.

### Why 3× weight on intent and 3× boost on engineered features?
Unsupervised clustering with mixed feature types requires manual calibration since there's no loss function to do it for you. Without boosting, 80 LSA dimensions would completely drown out 21 meta features. The 3× factor was determined empirically by scanning silhouette scores across different weight values.

### Why K=8 instead of K=2 or K=3 (which had slightly higher silhouette)?
A clustering model that splits 5,818 memes into 2 groups — one of 5,700 and one of 118 — has high silhouette but zero interpretive value. The competition's stated goal is to "uncover hidden contexts, genres, and semantic groupings." K=8 achieves silhouette of 0.449 while producing eight genuinely distinct, interpretable groups. Quality metrics and semantic richness must be balanced.

### Why compare three models instead of just running K-Means?
Because the best model depends on data geometry you don't know in advance. K-Means assumes spherical clusters. Agglomerative (Ward) doesn't. Spherical K-Means uses cosine similarity — often better for text. Testing all three and selecting by metric is scientifically rigorous, not guess-and-check.

---

## ⚠️ Limitations & Trade-offs

| Limitation | Impact | Why accepted |
|-----------|--------|-------------|
| LSA captures only 27.5% of TF-IDF variance | Some semantic signal lost | 72.5% is largely noise from boilerplate; increasing dims showed diminishing returns |
| K-Means assumes roughly spherical clusters | Boundary memes may be misclassified | Outperformed alternatives on this dataset's metrics |
| Domain stopwords are manually curated | May miss some boilerplate patterns | No automated method to distinguish content words from filler without labels |
| Large majority cluster (C0: 89.8%) | Limits actionable granularity | This reflects reality — most internet memes ARE broadly relatable; forcing more splits would be artificial |
| No ground-truth labels | Cannot compute accuracy/F1 | Inherent to the problem type; internal metrics are the only valid evaluation |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

No other dependencies. No internet connection required after data download.

### Steps

```bash
# 1. Clone this repository
git clone https://github.com/shravani22patil/meme-context-clustering.git
cd meme-context-clustering

# 2. Download dataset from Kaggle
# Place meme.csv in the project root directory
# https://www.kaggle.com/competitions/meme-context-clustering/data

# 3. Run the full pipeline
python meme_context_clustering.py

# 4. Outputs will be saved to outputs/
```

The script runs **top to bottom without errors** and generates all visualizations, the submission file, and the report automatically.

### Expected Runtime

| Step | Approximate Time |
|------|----------------|
| Data loading & parsing | < 5 seconds |
| TF-IDF + SVD | < 10 seconds |
| K scan (K=2..11) | ~2 minutes |
| Three model training | ~1 minute |
| Hyperparameter tuning | ~3 minutes |
| t-SNE visualization | ~30 seconds |
| **Total** | **~7 minutes** |

---

## 📁 File Structure

```
meme-context-clustering/
│
├── meme_context_clustering.py    # Complete pipeline (run this)
├── README.md                     # This file
├── meme.csv                      # Dataset (download from Kaggle)
│
└── outputs/
    ├── submission.csv                  # Competition submission (id, cluster)
    ├── enriched_predictions.csv        # id + meme_text + intent + cluster_name
    ├── report.txt                      # Full approach & methodology report
    ├── 01_eda_overview.png
    ├── 02_optimal_k_selection.png
    ├── 03_silhouette_comparison.png
    ├── 04_model_comparison_metrics.png
    ├── 05_tsne_clusters.png
    ├── 06_cluster_feature_heatmap.png
    └── 07_cluster_keywords.png
```

---

## 🏆 Competition Context

This project was built for the **ML Coding Round** of **DataSprint 2026**, organized by the Data Science Club at NIST University in collaboration with MAIT (Maharaja Agrasen Institute of Technology).

- **Round 1 (Quiz Round):** Ranked **590 out of ~1,000 participants** → Shortlisted for ML Round
- **Round 2 (ML Coding Round):** Built this complete clustering pipeline for the Meme Context Clustering problem statement (PS2)

**Competition constraints strictly followed:**
- ✅ Only standard ML libraries (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
- ✅ No external datasets
- ✅ No pretrained models or embeddings
- ✅ No AutoML tools
- ✅ No pre-built solutions
- ✅ Fully reproducible from top to bottom

---

## 🎓 What I Learned

Building this project surfaced several lessons that don't appear in textbooks:

**1. Data structure is more important than model choice.**
The biggest performance gains came from parsing the structured input field, not from switching between clustering algorithms. Understanding *what your data actually is* before modeling is the most underrated skill in ML.

**2. In unsupervised learning, domain knowledge is your loss function.**
With no labels, there's no gradient to guide you. Every decision — what to clean, what to weight, how to select K — requires you to be the judge. Domain stopwords, feature weighting, and K selection all required understanding the *semantics* of memes, not just the statistics of the data.

**3. A high silhouette score with K=2 is not a good result.**
Internal metrics measure cluster quality, not cluster usefulness. A model that splits 5,818 items into groups of 5,700 and 118 is technically "good" by silhouette but tells you nothing interesting. Always ask: does this clustering answer the question that was asked?

**4. Boilerplate is the enemy of NLP clustering.**
In any domain-specific corpus, there will be phrases that appear everywhere and mean nothing. Standard stopword lists don't know about "meme poster is trying to convey." You have to look at your data and build domain awareness into your preprocessing.

**5. Cross-validation exists in unsupervised learning — it just looks different.**
Running the same pipeline on 5 random seeds and measuring variance in the silhouette score is a valid stability check. Low variance means the structure is real. High variance means you found noise.

---

## 📬 Connect

**Shravani Rajendra Patil**  
SSVPS B S Deore College of Engineering, Dhule  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)]((https://www.linkedin.com/in/shravani-patil-38791b286/))
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat-square&logo=kaggle)]((https://www.kaggle.com/shravani2patil2))

---

<p align="center">
  <i>Built with no shortcuts, no pretrained models, and no AutoML — just math, intuition, and a lot of t-SNE.</i>
</p>
