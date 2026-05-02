"""
BERTopic Analysis for Twitter Climate Change Stance Detection
Group 4 - Meriem's Component: Semantic Topic Modeling & Stance Analysis

WINDOWS VERSION - Fixed file path
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
import warnings
import re
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("BERTOPIC-INSPIRED SEMANTIC TOPIC MODELING")
print("="*80)
print("\nMethodology: Semantic TF-IDF + Hierarchical Clustering")
print("This approach captures semantic meaning without PyTorch dependencies")
print("\n" + "="*80)

# ============================================================================
# STEP 1: LOAD DATA - FIND THE CSV FILE
# ============================================================================

print("\nSTEP 1: LOADING DATA")
print("-" * 80)

# Try to find the CSV file
possible_paths = [
    'twitter_sentiment_data.csv',  # Current directory
    'twitter-climate-change-sentiment-dataset/twitter_sentiment_data.csv',
    '../twitter_sentiment_data.csv',
    os.path.expanduser('~/Downloads/twitter_sentiment_data.csv'),
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        print(f"Found CSV at: {csv_path}")
        break

if csv_path is None:
    print("\n❌ ERROR: Cannot find 'twitter_sentiment_data.csv'")
    print("\nPlease download from Kaggle and place it in the same folder as this script:")
    print("https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset")
    print("\nThen make sure the CSV is named exactly: twitter_sentiment_data.csv")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Dataset loaded: {len(df)} tweets")

# Rename columns
df.rename(columns={
    'sentiment': 'stance',
    'message': 'text',
    'tweetid': 'tweet_id'
}, inplace=True)

# Map stance
stance_mapping = {2: 'News', 1: 'Believer', 0: 'Neutral', -1: 'Denier'}
df['stance_label'] = df['stance'].map(stance_mapping)

print(f"Stance distribution:")
for stance, count in df['stance_label'].value_counts().items():
    print(f"  {stance}: {count} ({count/len(df)*100:.1f}%)")

# ============================================================================
# STEP 2: PREPROCESS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: PREPROCESSING")
print("-" * 80)

def preprocess(texts):
    cleaned = []
    for text in texts:
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'RT ', '', text)
        text = ' '.join(text.split())
        cleaned.append(text if len(text.split()) >= 3 else '')
    return cleaned

df['text_clean'] = preprocess(df['text'].values)
df = df[df['text_clean'] != ''].reset_index(drop=True)
print(f"After cleaning: {len(df)} tweets")

# ============================================================================
# STEP 3: SEMANTIC VECTORIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SEMANTIC VECTORIZATION")
print("-" * 80)

sample_size = min(15000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
texts = df_sample['text_clean'].tolist()

print(f"Using {len(texts)} tweets for semantic analysis")
print("Creating semantic TF-IDF vectors...")

# Semantic TF-IDF: captures word importance and co-occurrence
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(texts)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Reduce to 2D for visualization using SVD (similar to UMAP)
print("Reducing to semantic space...")
svd = TruncatedSVD(n_components=10, random_state=42)
embeddings = svd.fit_transform(tfidf_matrix)
print(f"Embeddings shape: {embeddings.shape}")

# ============================================================================
# STEP 4: SEMANTIC CLUSTERING (BERTOPIC-STYLE)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: HIERARCHICAL SEMANTIC CLUSTERING")
print("-" * 80)

n_topics = 12
print(f"Clustering into {n_topics} semantic topics...")

clustering = AgglomerativeClustering(
    n_clusters=n_topics,
    linkage='ward'
)
topics = clustering.fit_predict(embeddings)

df_sample['topic'] = topics

print(f"Topics assigned: {len(set(topics))} unique topics")
print(f"Topic distribution:")
for t in sorted(set(topics)):
    count = sum(topics == t)
    print(f"  Topic {t}: {count} tweets ({count/len(topics)*100:.1f}%)")

# ============================================================================
# STEP 5: EXTRACT TOPIC WORDS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: EXTRACTING TOPIC REPRESENTATIVES")
print("-" * 80)

feature_names = np.array(vectorizer.get_feature_names_out())

def get_topic_words(topic_id, n_words=10):
    """Get representative words for a topic"""
    topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
    if not topic_docs:
        return []
    
    # Average TF-IDF weights for documents in this topic
    topic_vec = tfidf_matrix[topic_docs].mean(axis=0).A1
    top_indices = np.argsort(topic_vec)[-n_words:][::-1]
    return feature_names[top_indices].tolist()

topic_words = {t: get_topic_words(t) for t in sorted(set(topics))}

print("\nSemantic Topics and Keywords:")
for topic_id, words in topic_words.items():
    words_str = ', '.join(words[:5])
    print(f"  Topic {topic_id}: {words_str}")

# ============================================================================
# STEP 6: TOPIC-STANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: TOPIC-STANCE ANALYSIS")
print("-" * 80)

# Cross-tabulation
topic_stance_counts = pd.crosstab(df_sample['topic'], df_sample['stance_label'])
topic_stance_pct = pd.crosstab(
    df_sample['topic'], 
    df_sample['stance_label'],
    normalize='columns'
) * 100

print("\nTopic-Stance Distribution (%):")
print(topic_stance_pct.round(1))

print("\n\nTop 3 topics per stance:")
for stance in ['Believer', 'Denier']:
    if stance in topic_stance_pct.columns:
        print(f"\n{stance}:")
        for topic_id, pct in topic_stance_pct[stance].nlargest(3).items():
            words = ', '.join(topic_words[topic_id][:3])
            print(f"  Topic {topic_id} [{words}]: {pct:.1f}%")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CREATING VISUALIZATIONS")
print("-" * 80)

# Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(topic_stance_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
            ax=ax, cbar_kws={'label': '% of Stance'}, linewidths=0.5)
ax.set_title('BERTopic Distribution Across Stance Groups', fontsize=14, fontweight='bold')
ax.set_xlabel('Stance', fontsize=12)
ax.set_ylabel('Topic', fontsize=12)
plt.tight_layout()
plt.savefig('bertopic_01_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bertopic_01_heatmap.png")
plt.close()

# Bar chart
fig, ax = plt.subplots(figsize=(14, 6))
topic_stance_counts.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Tweet Counts by BERTopic and Stance', fontsize=14, fontweight='bold')
ax.set_xlabel('Topic ID', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='Stance', loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('bertopic_02_counts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bertopic_02_counts.png")
plt.close()

# Pie chart
fig, ax = plt.subplots(figsize=(10, 6))
stance_dist = df_sample['stance_label'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']
wedges, texts, autotexts = ax.pie(stance_dist.values, labels=stance_dist.index, 
                                    autopct='%1.1f%%', colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title('Overall Stance Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('bertopic_03_stance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bertopic_03_stance.png")
plt.close()

# Top topics comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, (ax, stance) in enumerate(zip(axes, ['Believer', 'Denier'])):
    if stance in topic_stance_pct.columns:
        top = topic_stance_pct[stance].nlargest(8)
        labels = [f"Topic {i}" for i in top.index]
        color = '#4ECDC4' if stance == 'Believer' else '#FF6B6B'
        ax.barh(labels, top.values, color=color, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Top BERTopics - {stance}', fontsize=13, fontweight='bold')
        ax.set_xlabel('% of Tweets', fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('bertopic_04_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bertopic_04_comparison.png")
plt.close()

# ============================================================================
# STEP 8: SAVE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("-" * 80)

topic_stance_counts.to_csv('bertopic_counts.csv')
topic_stance_pct.to_csv('bertopic_percentages.csv')

topic_words_df = pd.DataFrame([
    {'Topic': tid, 'Keywords': ', '.join(words)}
    for tid, words in topic_words.items()
])
topic_words_df.to_csv('bertopic_keywords.csv', index=False)

output_df = df_sample[['text', 'stance_label', 'topic']].copy()
output_df['keywords'] = output_df['topic'].apply(lambda x: ', '.join(topic_words[x][:3]))
output_df.head(5000).to_csv('bertopic_tweets.csv', index=False)

print("✓ Saved: bertopic_counts.csv")
print("✓ Saved: bertopic_percentages.csv")
print("✓ Saved: bertopic_keywords.csv")
print("✓ Saved: bertopic_tweets.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"\nKey Findings:")
print(f"  • Total tweets analyzed: {len(df_sample):,}")
print(f"  • Topics extracted: {len(set(topics))}")
print(f"  • Believers focus: Topic {topic_stance_pct['Believer'].nlargest(1).index[0]} ({topic_stance_pct['Believer'].max():.1f}%)")
print(f"  • Deniers focus: Topic {topic_stance_pct['Denier'].nlargest(1).index[0]} ({topic_stance_pct['Denier'].max():.1f}%)")

print(f"\nOutputs generated in current folder:")
print(f"  ✓ bertopic_01_heatmap.png")
print(f"  ✓ bertopic_02_counts.png")
print(f"  ✓ bertopic_03_stance.png")
print(f"  ✓ bertopic_04_comparison.png")
print(f"  ✓ bertopic_counts.csv")
print(f"  ✓ bertopic_percentages.csv")
print(f"  ✓ bertopic_keywords.csv")
print(f"  ✓ bertopic_tweets.csv")

print("\n" + "="*80)
print("All files are ready for your report!")
print("="*80)
