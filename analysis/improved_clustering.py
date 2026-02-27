import pandas as pd
import numpy as np
import re
import os
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from pathlib import Path

# --- Configuration ---
# Get the script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Use relative paths from project root
INPUT_FILE = PROJECT_ROOT / 'outputs' / 'leiden_clustering' / 'Paddy (Dhan)' / 'mapping.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'leiden_clustering' / 'Paddy (Dhan)'
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
ALPHA = 0.5  # Weight for Dense Embeddings (0.5 Dense, 0.5 Sparse)

# Expanded Stop Words List (Hinglish + English + General Ag Domain)
# Removed specific crop names to make it robust for any crop analysis
STOP_WORDS = set([
    # English equivalents/Articles/Prepositions
    'a', 'an', 'the', 'in', 'on', 'of', 'for', 'to', 'at', 'by', 'from', 'with', 'and', 'or', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'shoud', 'would', 'will', 'may', 'might', 'must',
    
    # Hinglish Structural/Grammar
    'mein', 'me', 'mai', 'main', 'ko', 'ka', 'ki', 'ke', 'se', 'ne', 'par', 'per', 'pe', 'he', 'hai', 'hain', 'ha', 'hi', 'ho', 'h', 'tha', 'thi', 'the', 'hu', 'hun', 'hoon', 'kya', 'kyon', 'kab', 'kaise', 'kaha', 'kahan', 'kon', 'kaun', 'kis', 'kisi', 'kisko', 'kisse',
    'liye', 'liya', 'lie', 'laye', 'wale', 'wali', 'wala', 'aur', 'tatha', 'evam', 'bhi', 'to', 'agar', 'magar', 'lekin', 'parantu',

    # Action Verbs (ignoring tense/form)
    'kare', 'karen', 'karein', 'karna', 'karne', 'kar', 'karo', 'kiya', 'kiye', 'jay', 'jaye', 'jana', 'jane', 'jata', 'jati', 'jate',
    'lag', 'laga', 'lagi', 'lage', 'lga', 'lgi', 'lge', 'lagna', 'lagne', 'aa', 'aaya', 'aaye', 'raha', 'rahi', 'rahe', 'rha', 'rhi', 'rhe',
    'ho', 'hona', 'hone', 'honi', 'pad', 'pd', 'padi', 'pdi', 'padta', 'padti', 'padte', 'gaya', 'gyi', 'gaye', 'gye', 'gay', 'gayi',
    'chahiye', 'chahie', 'chahye', 'sakta', 'sakti', 'sakte', 'sake', 'saka',
    'bataye', 'batayen', 'batain', 'batao', 'bataiye', 'btaye', 'btao', 'bata', 'de', 'den', 'dijiye', 'dijiya', 'dein',
    'dal', 'dala', 'dale', 'dalen', 'dalna', 'dalne', 'dali', 'dalo', 'use', 'using', 'used', 'apply', 'applied', 'applying', 'spray', 'spraying',

    # Generic Ag Domain Words (Crop Agnostic)
    'crop', 'phasal', 'fasal', 'kheti', 'farm', 'farming', 'farmer', 'kisan', 'agriculture', 'krishi',
    'plant', 'paudha', 'podha', 'paudh', 'podh', 'ped', 'field', 'khet', 'seed', 'beej', 'seeds',
    'variety', 'var', 'varieties', 'kism', 'prajati', 'prajatiyan', 'type',
    # Specific crop names should be added dynamically or we keep a list of common ones if we want to suppress them
    # For now, suppressing generic "dhan/paddy" if specific to this script, but user asked for robust.
    # We will add a list of COMMON crop names to suppress as they are usually the *context* not the *problem*.
    'dhan', 'paddy', 'rice', 'chawal', 'gehu', 'wheat', 'ganna', 'sugarcane', 'makka', 'maize', 'bajra', 'millet', 'aloo', 'potato', 'tamatar', 'tomato',
    'ji', 'sir', 'madam', 'mam', 'mr', 'bhai', 'bhaiya', 

    # High Frequency Fillers
    'question', 'query', 'problem', 'samasya', 'issue', 'doubt', 'help', 'info', 'information', 'jankari', 'detail', 'details',
    'about', 'regarding', 'related', 'want', 'know', 'please', 'pls', 'plz', 'provide', 'tell', 'ask', 'asked', 'asking',
    'solution', 'upchar', 'ilaaj', 'ilaj', 'dawa', 'medicine', 'control', 'roktham', 'nidan', 'upay', 'management', 'niyantran',
    'matra', 'quantity', 'dose', 'amount', 'rate', 'price', 'bhav', 'rate', 'muly',
    'number', 'no', 'contact', 'mobile',
]) 

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars and digits (keep only alphabets and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove Stopwords for Embedding Separation
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    
    return " ".join(tokens)

def extract_keywords_tfidf(texts, stop_words, max_features=1000, max_df=0.95, min_df=2):
    """
    Extracts top keywords for each text using TF-IDF, biased against stop words.
    """
    print("Extracting TF-IDF keywords...")
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=list(stop_words), max_df=max_df, min_df=min_df)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError: # Handle case with empty vocab
        print("Warning: TF-IDF failed, possibly due to stop words. Using empty keywords.")
        return [], None, None

    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get top keywords for each document
    top_keywords = []
    
    # Iterate over rows
    # tfidf_matrix is CSR sparse matrix
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i]
        _, col_indices = row.nonzero()
        
        if len(col_indices) == 0:
            top_keywords.append(set())
            continue
            
        scores = row.data
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = col_indices[sorted_indices[:5]] 
        
        keywords = set(feature_names[top_k_indices])
        top_keywords.append(keywords)
        
    return top_keywords, tfidf_matrix, vectorizer

def calculate_hybrid_distance(dense_embeddings, tfidf_matrix, alpha=0.7):
    """
    Calculates hybrid distance: alpha * cosine_dist + (1-alpha) * jaccard_dist
    """
    print("Calculating Hybrid Distance...")
    
    # 1. Dense Distance
    print("  - Dense Distance...")
    dense_dist = pairwise_distances(dense_embeddings, metric='cosine', n_jobs=-1)
    
    # 2. Sparse Distance (Jaccard Distance on TF-IDF binary presence)
    print("  - Sparse Distance...")
    tfidf_bool = (tfidf_matrix > 0).astype(bool)
    # Convert to dense for pairwise_distances (safe for <50k rows)
    try:
        tfidf_dense_bool = tfidf_bool.toarray()
        sparse_dist = pairwise_distances(tfidf_dense_bool, metric='jaccard', n_jobs=-1)
    except MemoryError:
        print("Memory Error in Sparse Distance Calculation. Falling back to Dense only for this step (not ideal but safe).")
        return dense_dist
    
    # 3. Weighted Average
    print("  - Fusing Matrices...")
    hybrid_dist = alpha * dense_dist + (1 - alpha) * sparse_dist
    
    return hybrid_dist

def main():
    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
    
    # Fill NaN
    df['query_text'] = df['query_text'].fillna("")
    
    # Preprocess
    print("Preprocessing...")
    df['cleaned_text'] = df['query_text'].apply(preprocess_text)
    
    # Remove empty
    df = df[df['cleaned_text'].str.strip().astype(bool)].copy()
    print(f"Processing {len(df)} valid queries...")

    # Extract Keywords
    top_keywords_list, tfidf_matrix, vectorizer = extract_keywords_tfidf(df['cleaned_text'], STOP_WORDS)
    df['top_keywords'] = [", ".join(list(k)) for k in top_keywords_list]
    
    # Generate Embeddings
    print(f"Loading Sentence Transformer {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings...")
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=32)
    
    # Calculate Hybrid Distance Matrix
    hybrid_dist_matrix = calculate_hybrid_distance(embeddings, tfidf_matrix, alpha=ALPHA)
    
    # Dimensionality Reduction (UMAP)
    print("Reducing dimensions with UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.0, 
        n_components=5, 
        metric='precomputed', 
        random_state=42
    )
    # UMAP expects square matrix for 'precomputed'
    reduced_data = umap_model.fit_transform(hybrid_dist_matrix)
    
    # Clustering (HDBSCAN)
    print("Clustering with HDBSCAN...")
    # Tweak these parameters for cluster granularity
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3, # Lowered to catch small distinct groups
        min_samples=2,      # Lowered to allow tighter clusters
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    labels = clusterer.fit_predict(reduced_data)
    
    # --- Noise Reassignment ---
    print("Handling Noise...")
    # 1. High Volume Noise -> New Clusters
    # We need to access counts. 
    counts = df['count'].values
    unique_labels = np.unique(labels)
    max_label = unique_labels.max()
    
    noise_indices = np.where(labels == -1)[0]
    
    print(f"Initial Noise Count: {len(noise_indices)}")
    
    for idx in noise_indices:
        if counts[idx] > 50: # If a single query appearing typically > 50 times is noise, make it a cluster
            max_label += 1
            labels[idx] = max_label
            
    # 2. Remaining Noise -> Nearest Neighbor
    # Fit KNN on non-noise
    from sklearn.neighbors import KNeighborsClassifier
    non_noise_mask = labels != -1
    
    if non_noise_mask.sum() > 0:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(reduced_data[non_noise_mask], labels[non_noise_mask])
        
        remaining_noise_indices = np.where(labels == -1)[0]
        if len(remaining_noise_indices) > 0:
            print(f"Reassigning {len(remaining_noise_indices)} remaining noise points...")
            predicted_labels = knn.predict(reduced_data[remaining_noise_indices])
            labels[remaining_noise_indices] = predicted_labels
            
    df['cluster_id'] = labels
    
    # Save Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_mapping_file = os.path.join(OUTPUT_DIR, 'mapping_improved.csv')
    print(f"Saving results to {output_mapping_file}...")
    
    # Ensure 'count' exists
    if 'count' not in df.columns:
        df['count'] = 1
        
    cols = ['query_text', 'cluster_id', 'cleaned_text', 'top_keywords', 'count']
    df[cols].to_csv(output_mapping_file, index=False)
    
    # Generate Summary
    print("Generating summary...")
    summary_data = []
    grouped = df.groupby('cluster_id')
    
    for cluster_id, group in grouped:
        if cluster_id == -1:
            continue
            
        unique_queries = len(group)
        total_volume = group['count'].sum()
        
        # Representative Query
        rep_query = group.sort_values('count', ascending=False).iloc[0]['query_text']
        
        # Top Keywords for cluster
        all_keywords = []
        for k_str in group['top_keywords']:
            if k_str:
                all_keywords.extend([k.strip() for k in k_str.split(',')])
        
        from collections import Counter
        if all_keywords:
            common_keywords = [k for k, v in Counter(all_keywords).most_common(5)]
        else:
            common_keywords = []
        
        summary_data.append({
            'cluster_id': cluster_id,
            'size': total_volume,
            'unique_queries': unique_queries,
            'representative': rep_query,
            'top_keywords': ", ".join(common_keywords)
        })
        
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('size', ascending=False)
        output_summary_file = os.path.join(OUTPUT_DIR, 'summary_improved.csv')
        summary_df.to_csv(output_summary_file, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
