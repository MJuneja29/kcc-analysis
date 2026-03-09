import pandas as pd
import numpy as np
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import hdbscan
import umap

class ClusteringPipeline:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 alpha=0.5, algorithm='hdbscan',
                 algorithm_params=None, use_umap=True, umap_n_components=5,
                 use_char_features=False, char_ngram_range=(2, 5),
                 use_length_feature=False, crop_name=None, metadata_columns=None):
        """
        Initializes the Clustering Pipeline.
        
        :param model_name: HuggingFace model name for SentenceTransformer
        :param alpha: Weight for dense embeddings in hybrid distance (0 to 1). 1.0 = 100% dense.
        :param algorithm: 'hdbscan', 'kmeans', 'agglomerative', etc.
        :param algorithm_params: Dictionary of kwargs to pass to the clustering algorithm
        :param use_umap: Whether to use UMAP dimensionality reduction before clustering
        :param umap_n_components: Number of UMAP components to reduce to
        :param use_char_features: Whether to extract Character n-gram features alongside Word TF-IDF
        :param char_ngram_range: Tuple for char n-gram bounds (min, max)
        :param use_length_feature: Whether to add query character length as a normalized dense feature
        :param crop_name: Optional string to prepend to text for LLM embedding context (e.g., 'Crop: Wheat')
        :param metadata_columns: List of columns in df to One-Hot Encode and fuse
        """
        self.model_name = model_name
        self.alpha = float(alpha)
        self.algorithm = algorithm.lower()
        self.algorithm_params = algorithm_params or {}
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        
        self.use_char_features = use_char_features
        self.char_ngram_range = char_ngram_range
        self.use_length_feature = use_length_feature
        self.crop_name = crop_name
        self.metadata_columns = metadata_columns or []
        
        self.stop_words = self._get_stop_words()
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.char_vectorizer = None
        self.metadata_encoder = None
        self.length_scaler = None

    def _get_stop_words(self):
        # Base stop words from previous implementation + Hinglish additions
        return set([
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
            
            # Common crops to ignore context variations
            'dhan', 'paddy', 'rice', 'chawal', 'gehu', 'wheat', 'ganna', 'sugarcane', 'makka', 'maize', 'bajra', 'millet', 'aloo', 'potato', 'tamatar', 'tomato',
            'ji', 'sir', 'madam', 'mam', 'mr', 'bhai', 'bhaiya', 'namaste',

            # High Frequency Fillers
            'question', 'query', 'problem', 'samasya', 'issue', 'doubt', 'help', 'info', 'information', 'jankari', 'detail', 'details',
            'about', 'regarding', 'related', 'want', 'know', 'please', 'pls', 'plz', 'provide', 'tell', 'ask', 'asked', 'asking',
            'solution', 'upchar', 'ilaaj', 'ilaj', 'dawa', 'medicine', 'control', 'roktham', 'nidan', 'upay', 'management', 'niyantran',
            'matra', 'quantity', 'dose', 'amount', 'rate', 'price', 'bhav', 'rate', 'muly',
            'number', 'no', 'contact', 'mobile',
        ])
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Keep only alphabetic chars
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        return " ".join(tokens)

    def extract_sparse_features(self, texts):
        # Allow n-grams for feature engineering benchmark
        ngram_range = self.algorithm_params.get('ngram_range', (1, 1))
        
        from scipy.sparse import csr_matrix, hstack
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, max_df=0.95, min_df=2, ngram_range=ngram_range)
        try:
            word_matrix = self.tfidf_vectorizer.fit_transform(texts)
        except ValueError:
            word_matrix = csr_matrix((len(texts), 0))
            
        if self.use_char_features:
            self.char_vectorizer = TfidfVectorizer(
                analyzer='char', 
                ngram_range=self.char_ngram_range, 
                max_features=1000, 
                max_df=0.95, 
                min_df=5
            )
            try:
                char_matrix = self.char_vectorizer.fit_transform(texts)
                # Combine word and char features
                return hstack([word_matrix, char_matrix], format='csr')
            except ValueError:
                pass # If it fails (e.g., all texts tiny), just return word matrix
                
        return word_matrix

    def extract_length_feature(self, texts):
        lengths = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
        from sklearn.preprocessing import StandardScaler
        self.length_scaler = StandardScaler()
        return self.length_scaler.fit_transform(lengths)

    def extract_metadata_features(self, df, valid_indices):
        if not self.metadata_columns:
            return None
        metadata_df = df.loc[valid_indices, self.metadata_columns].copy()
        for col in metadata_df.columns:
            # Convert all to strings and fill NaN
            metadata_df[col] = metadata_df[col].fillna("Unknown").astype(str)
            
        from sklearn.preprocessing import OneHotEncoder
        self.metadata_encoder = OneHotEncoder(handle_unknown='ignore')
        return self.metadata_encoder.fit_transform(metadata_df)

    def generate_embeddings(self, texts):
        if self.crop_name:
            # Prepend context to force LLM into agricultural semantic space
            texts = [f"Crop: {self.crop_name}, Query: {t}" for t in texts]
            
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.model_name)
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings

    def concatenate_features(self, dense_embeddings, tfidf_matrix, length_features=None, metadata_features=None):
        """
        Early Fusion: Cancatenates Dense and Sparse features to avoid O(N^2) distance matrices.
        Scales the inputs based on the alpha parameter before joining.
        """
        if self.alpha == 1.0:
            base_features = dense_embeddings
        elif self.alpha == 0.0:
            base_features = tfidf_matrix
        else:
            # Hybrid Early Fusion
            dense_norm = normalize(dense_embeddings, norm='l2', axis=1)
            sparse_norm = normalize(tfidf_matrix, norm='l2', axis=1)
            
            dense_scaled = dense_norm * np.sqrt(self.alpha)
            sparse_scaled = sparse_norm * np.sqrt(1.0 - self.alpha)
            
            dense_sparse = sp.csr_matrix(dense_scaled)
            base_features = sp.hstack([dense_sparse, sparse_scaled], format='csr')

        # Accumulate all feature blocks
        pieces = [base_features if sp.issparse(base_features) else sp.csr_matrix(base_features)]
        
        if length_features is not None:
            # Weight length fairly lightly to avoid swamping semantic features
            pieces.append(sp.csr_matrix(length_features * 0.1))
            
        if metadata_features is not None:
            # Weight one-hot categorical metadata (gives a structural bump to identical states/seasons/crops)
            pieces.append(metadata_features * 0.25)
            
        # Combine into a single matrix
        combined_features = sp.hstack(pieces, format='csr')
        return combined_features

    def fit_predict(self, df, text_column='query_text', counts_column='count'):
        """
        Runs the full end-to-end pipeline and assigns a 'cluster_id' to existing DF.
        Returns the modified dataframe and the core feature matrix (for evaluation).
        """
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
        
        # Valid indices only
        valid_mask = df['cleaned_text'].str.strip().astype(bool)
        valid_indices = df.index[valid_mask].tolist()
        texts = df.loc[valid_indices, 'cleaned_text'].tolist()
        
        if len(texts) == 0:
            df['cluster_id'] = -1
            return df, None
            
        print(f"Extracted {len(texts)} valid texts out of {len(df)}.")
        
        # 1. TF-IDF
        print("\n[Step 1] Extracting TF-IDF Features...")
        t0 = time.time()
        tfidf_matrix = self.extract_sparse_features(texts)
        print(f"  ✓ TF-IDF Done in {time.time() - t0:.2f}s")
        
        # 2. Embeddings
        print("\n[Step 2] Generating Dense LLM Embeddings...")
        t0 = time.time()
        dense_embeddings = self.generate_embeddings(texts)
        print(f"  ✓ Embeddings Done in {time.time() - t0:.2f}s")
        
        # 2.5 New Features Extraction
        length_features = None
        if self.use_length_feature:
            print("\n[Step 2.5a] Extracting Length Features...")
            length_features = self.extract_length_feature(texts)
            
        metadata_features = None
        if self.metadata_columns:
            print(f"\n[Step 2.5b] Extracting Metadata Features for {self.metadata_columns}...")
            metadata_features = self.extract_metadata_features(df, valid_indices)
        
        # 3. Early Fusion
        print(f"\n[Step 3] Fusing Features (Alpha: {self.alpha})...")
        t0 = time.time()
        combined_features = self.concatenate_features(
            dense_embeddings, 
            tfidf_matrix, 
            length_features=length_features, 
            metadata_features=metadata_features
        )
        print(f"  ✓ Early Fusion Done in {time.time() - t0:.2f}s. Shape: {combined_features.shape}")
        
        clustering_input = combined_features
        
        if self.use_umap:
            # Algorithms need dense numpy arrays, not sparse CSR (unless specifically supported)
            if sp.issparse(clustering_input):
                 input_for_clustering = clustering_input.toarray()
            else:
                 input_for_clustering = clustering_input
                 
            print("\n[Step 4] Running UMAP Dimensionality Reduction...")
            t0 = time.time()
            umap_model = umap.UMAP(
                n_neighbors=self.algorithm_params.get('umap_neighbors', 15), 
                min_dist=0.0, 
                n_components=self.umap_n_components, 
                metric='cosine',
                random_state=42,
                verbose=True,                          # prints stage-by-stage progress
                tqdm_kwds={'desc': 'UMAP', 'leave': False}  # progress bar during layout optimisation
            )
            try:
                clustering_input = umap_model.fit_transform(input_for_clustering)
                print(f"  ✓ UMAP Done in {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"  x UMAP Failed. Falling back to raw features. Error: {e}")
                # Don't overwrite clustering_input if UMAP fails, leave it as sparse/dense features
        
        print(f"\n[Step 5] Running {self.algorithm.upper()} Clustering...")
        t0 = time.time()
        labels = np.array([-1] * len(texts))
        
        # Convert to dense for clustering algorithms if UMAP didn't already
        if sp.issparse(clustering_input):
            input_for_clustering = clustering_input.toarray()
        else:
            input_for_clustering = clustering_input
        
        if self.algorithm == 'hdbscan':
            min_cluster_size = self.algorithm_params.get('min_cluster_size', 3)
            min_samples = self.algorithm_params.get('min_samples', 2)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean', # We pass features or UMAP coords, not distances
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(input_for_clustering)
            
        elif self.algorithm == 'kmeans':
            n_clusters = self.algorithm_params.get('n_clusters', 2000)
            n_clusters = min(n_clusters, len(texts) - 1)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = clusterer.fit_predict(input_for_clustering)
            
        elif self.algorithm == 'agglomerative':
            n_clusters = self.algorithm_params.get('n_clusters', 2000)
            n_clusters = min(n_clusters, len(texts) - 1)
            linkage = self.algorithm_params.get('linkage', 'ward')
            if linkage == 'ward': 
                metric = 'euclidean'
            else:
                metric = 'cosine' # Default to cosine for text features if not Ward
            
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric=metric,
                linkage=linkage
            )
            labels = clusterer.fit_predict(input_for_clustering)

        elif self.algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            eps = self.algorithm_params.get('eps', 0.5)
            min_samples = self.algorithm_params.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(input_for_clustering)

        elif self.algorithm == 'optics':
            from sklearn.cluster import OPTICS
            min_samples = self.algorithm_params.get('min_samples', 15)
            clusterer = OPTICS(min_samples=min_samples)
            labels = clusterer.fit_predict(input_for_clustering)

        elif self.algorithm == 'gmm':
            from sklearn.mixture import GaussianMixture
            n_components = self.algorithm_params.get('n_components', 500)
            n_components = min(n_components, len(texts) - 1)
            clusterer = GaussianMixture(n_components=n_components, random_state=42)
            labels = clusterer.fit_predict(input_for_clustering)

        elif self.algorithm == 'spectral':
            from sklearn.cluster import SpectralClustering
            n_clusters = self.algorithm_params.get('n_clusters', 50) # kept low since O(N^3)
            n_clusters = min(n_clusters, len(texts) - 1)
            # Spectral is O(N^3), requires much memory on large N.
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, n_init=1)
            labels = clusterer.fit_predict(input_for_clustering)

        elif self.algorithm in ['leiden', 'louvain']:
            from sklearn.neighbors import kneighbors_graph
            import networkx as nx
            try:
                import community.community_louvain as community_louvain
                print("  -> Building KNN Graph for Louvain Community Detection...")
                A = kneighbors_graph(input_for_clustering, n_neighbors=15, mode='distance')
                G = nx.from_scipy_sparse_array(A)
                partition = community_louvain.best_partition(G)
                labels = np.array([partition.get(i, -1) for i in range(len(texts))])
            except ImportError:
                print("Warning: python-louvain not installed. Falling back to HDBSCAN.")
                clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
                labels = clusterer.fit_predict(input_for_clustering)
            
        print(f"  ✓ Clustering Done in {time.time() - t0:.2f}s. Found {len(set(labels))} clusters.")
            
        # Optional: Handle noise via nearest neighbor re-assignment
        # We might want to disable this during benchmarking to see pure algorithm power
        handle_noise = self.algorithm_params.get('handle_noise', False)
        if handle_noise and -1 in labels and len(set(labels)) > 1:
            print("Handling noise points...")
            counts = df.loc[valid_indices, counts_column].values if counts_column in df.columns else np.ones(len(texts))
            unique_labels = np.unique(labels)
            max_label = unique_labels.max()
            
            noise_indices = np.where(labels == -1)[0]
            
            # Sub-cluster heavy noise elements
            for idx in noise_indices:
                if counts[idx] > 50:
                    max_label += 1
                    labels[idx] = max_label
            
            # KNN for remaining noise
            non_noise_mask = labels != -1
            if non_noise_mask.sum() > 0:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(input_for_clustering[non_noise_mask], labels[non_noise_mask])
                
                remaining_noise_indices = np.where(labels == -1)[0]
                if len(remaining_noise_indices) > 0:
                    predicted_labels = knn.predict(input_for_clustering[remaining_noise_indices])
                    labels[remaining_noise_indices] = predicted_labels
                    
        # Re-assign labels cleanly back to full dataframe
        df['cluster_id'] = -1
        df.loc[valid_indices, 'cluster_id'] = labels
        
        return df, clustering_input, combined_features
