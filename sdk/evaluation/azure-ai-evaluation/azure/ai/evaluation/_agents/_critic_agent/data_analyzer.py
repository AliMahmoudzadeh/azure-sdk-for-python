"""
General purpose DataAnalyzer for clustering conversational/contextual entries.

Input: List of entries, each containing:
    - context: str | dict | list (background info, task instructions, etc.)
    - conversation: str | dict | list (dialogue turns) – optional
    - metadata: dict (arbitrary attributes) – optional

Processing steps:
1. Summarize (textual) each entry focusing first on context, then (optionally) conversation.
2. Generate a concise, snake_case subcluster label and a short explanation (subcluster_description) for each subcluster, emphasizing CONTEXT meaning.
3. Cluster subcluster labels into up to N higher-level clusters via LLM (or heuristic fallback), using subcluster descriptions to generate cluster label and description.
4. Record conversation turn count in metadata as 'conversation_turns'.

Output structure (drill-down friendly):
{
  'summary': {
      'total_entries': int,
      'unique_subcluster_labels': int,
      'total_clusters': int,
      'clustering_method': str,
  },
  'entries': [
      {
        'id': <id or index>,
        'context_summary': str,
        'conversation_summary': str | None,
        'combined_summary': str,
        'subcluster_label': str,
        'subcluster_description': str,
        'coordinates': [float, float],
        'metadata': dict
      }, ...
  ],
  'subclusters': {
      '<subcluster_label>': {
          'entry_ids': [...],
          'count': int,
          'coordinates': [float, float],
      }, ...
  },
  'clusters': {
       '<cluster_name>': {
            'weight': int,
            'subcluster_labels': [...],
            'subcluster_counts': { label: count },
            'description': str,
            'coordinates': [float, float],
       }, ...
  },
  'clustering_quality': {
      'silhouette_score': float | None,
      'calinski_harabasz_score': float | None,
      'davies_bouldin_score': float | None,
      'inertia_score': float | None,
      'cluster_balance': float | None,
      'noise_points': int | None,  # HDBSCAN only
      'noise_ratio': float | None,  # HDBSCAN only
      'cluster_persistence': float | None,  # HDBSCAN only
      'quality_summary': str
  },
  'llm_analysis': str | None,
  'axes': [str, str],
  'raw': <original input list>,
  'classic_metadata': dict (optional)
}

This mirrors the ErrorAnalyzer pattern but is domain-agnostic and now includes subcluster and cluster explanations.
Supports both K-means and HDBSCAN clustering algorithms.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import os
import json
from collections import defaultdict, Counter as _Counter
import random
import pickle
import re
import math
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan  # type: ignore

from dotenv import load_dotenv
try:
    from openai import AzureOpenAI  # type: ignore
except Exception:
    AzureOpenAI = None  # type: ignore
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.decomposition import TruncatedSVD  # type: ignore
except Exception:
    TfidfVectorizer = None
    TruncatedSVD = None

import plotly.express as px
import pandas as pd

EntryType = Dict[str, Any]


class DataAnalyzer:
    """Analyze arbitrary (context + conversation + metadata) entries to produce hierarchical clusters."""

    def __init__(self, openai_client: Any = None):
        load_dotenv()
        self.openai_client = openai_client
        if self.openai_client is None and AzureOpenAI is not None:
            try:
                self.openai_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                )
            except Exception:
                self.openai_client = None
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    # =============================================================
    # Public API
    # =============================================================
    def process_entries(
        self,
        entries: List[EntryType] = [],
        structured: bool = True,
        input_files: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Preprocess raw entries into enriched processed entries with summaries and embeddings.

        Step 1 in workflow. Call this first, then pass the returned list to `analyze`.

        Parameters
        ----------
        entries : list
            Raw entries already in memory. Each should have at least a 'context' key.
        structured : bool
            If True, treat context as already concise (skip LLM summarization for context only).
        input_files : list[str] | None
            Optional JSON / JSONL files to load and merge with provided entries.
        save_path : str | None
            If provided, writes the processed entries JSON to this path.

        Returns
        -------
        list[dict]
            Processed entries each containing: id, context_summary, conversation_summary, combined_summary,
            metadata (enriched), embeddings (vector list[float]).
        """
        if input_files:
            loaded_from_files: List[EntryType] = []
            for path in input_files:
                try:
                    loaded = self._load_entries_from_file(path)
                    if loaded:
                        loaded_from_files.extend(loaded)
                except Exception as exc:  # noqa: BLE001
                    loaded_from_files.append(
                        {
                            "context": f"_file_load_error: {path}",
                            "metadata": {"error": str(exc)},
                        }
                    )
            if loaded_from_files:
                entries = list(entries) + loaded_from_files

        processed_entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            context = entry.get("context")
            conversation = entry.get("conversation")
            metadata = entry.get("metadata", {})
            entry_id = entry.get("id", idx)

            if structured:
                context_summary = str(context)
            else:
                context_summary = self._summarize_text(context, focus="context")
            conversation_summary = self._summarize_text(conversation, focus="conversation") if conversation else None
            combined_summary = self._combine_summaries(context_summary, conversation_summary)

            turns = self._count_conversation_turns(conversation)
            entry_time = None
            if isinstance(conversation, dict):
                entry_time = conversation.get("query", [{}])[0].get("createdAt") if conversation.get("query") else None
            enriched_metadata = dict(metadata)
            enriched_metadata.setdefault("conversation_turns", turns)
            if entry_time is not None:
                enriched_metadata.setdefault("entry_time", entry_time)

            processed_entries.append(
                {
                    "id": entry_id,
                    "context_summary": context_summary,
                    "conversation_summary": conversation_summary,
                    "combined_summary": combined_summary,
                    "metadata": enriched_metadata,
                    "conversation": conversation
                }
            )

        # Create embeddings if missing
        if processed_entries and any("embeddings" not in e for e in processed_entries):
            X, _ = self._create_embeddings(processed_entries)
            for i, x in enumerate(X):
                processed_entries[i]["embeddings"] = x.tolist() if hasattr(x, "tolist") else list(x)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(processed_entries, f)
        return processed_entries

    def analyze(
        self,
        processed_entries: List[Dict[str, Any]],
        num_clusters: int = 8,
        classic_subcluster_k: Optional[int] = None,
        clustering_method: str = "kmeans",
    ) -> Dict[str, Any]:
        """Cluster & label already processed entries (Step 2).

        Parameters
        ----------
        processed_entries : list
            Output from `process_entries` containing summaries & embeddings.
        num_clusters : int
            Desired maximum number of top-level clusters (KMeans path).
        classic_subcluster_k : int | None
            Optional override for KMeans subcluster count.
        clustering_method : str
            'kmeans' or 'hdbscan'
        """
        if not processed_entries:
            return {
                "summary": {
                    "total_entries": 0,
                    "unique_subcluster_labels": 0,
                    "total_clusters": 0,
                    "clustering_method": clustering_method,
                },
                "entries": [],
                "subclusters": {},
                "clusters": {},
                "clustering_quality": {},
                "llm_analysis": None,
                "axes": ["embeddings1", "embeddings2"],
                "raw": [],
            }

        if any("embeddings" not in e for e in processed_entries):
            raise ValueError("All processed entries must include 'embeddings'. Run process_entries() first or attach embeddings.")

        subclusters, clusters, clustering_quality, extra = self._perform_clustering(
            processed_entries,
            num_clusters=num_clusters,
            subcluster_k=classic_subcluster_k,
            clustering_method=clustering_method,
        )

        llm_analysis = self._analyze_overall(processed_entries, subclusters, clusters)
        axis_names = ["embeddings1", "embeddings2"]
        report = {
            "summary": {
                "total_entries": len(processed_entries),
                "unique_subcluster_labels": len(subclusters),
                "total_clusters": len(clusters),
                "clustering_method": clustering_method,
            },
            "entries": processed_entries,
            "subclusters": subclusters,
            "clusters": clusters,
            "clustering_quality": clustering_quality,
            "llm_analysis": llm_analysis,
            "axes": axis_names,
            "raw": processed_entries,
        }
        if extra:
            report["classic_metadata"] = extra
        return report

    # =============================================================
    # Orchestration / High-level analysis helpers
    # =============================================================
    def _perform_clustering(
        self,
        processed_entries: List[Dict[str, Any]],
        num_clusters: int,
        subcluster_k: Optional[int] = None,
        clustering_method: str = "kmeans",
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # Reuse existing embeddings from processed entries (avoid recomputation)
        embeddings = np.array([np.array(e["embeddings"]) for e in processed_entries])
        subclusters, clustering_quality = self._subcluster(
            embeddings, processed_entries, num_clusters, subcluster_k=subcluster_k, clustering_method=clustering_method
        )
        clusters = self._cluster(
            embeddings, processed_entries, subclusters, num_clusters, clustering_method=clustering_method
        )
        extra = {
            "method": "classic",
            "subcluster_k": subcluster_k,
            "embedding": "sbert",
            "clustering_algorithm": clustering_method,
            "top_level_clustering_algorithm": clustering_method,
            "dimensionality_reduction": "tsne",
            "warnings": [],
        }

        if clustering_method.lower() == "hdbscan":
            extra["hdbscan_noise_points"] = clustering_quality.get("noise_points", 0)
            extra["hdbscan_noise_ratio"] = clustering_quality.get("noise_ratio", 0.0)
            if clustering_quality.get("cluster_persistence") is not None:
                extra["hdbscan_avg_persistence"] = clustering_quality["cluster_persistence"]
        return subclusters, clusters, clustering_quality, extra

    def _analyze_overall(
        self,
        processed_entries: List[Dict[str, Any]],
        subclusters: Dict[str, Dict[str, Any]],
        clusters: Dict[str, Any],
    ) -> Optional[str]:
        if not self.openai_client or not self.deployment_name:
            return None
        overview = {
            "num_entries": len(processed_entries),
            "num_subclusters": len(subclusters),
            "num_clusters": len(clusters),
            "top_subclusters": sorted(
                ((k, v["count"]) for k, v in subclusters.items()), key=lambda x: x[1], reverse=True
            )[:10],
        }
        prompt = f"""
        Provide an insight summary of the thematic structure of the data based on this overview. Focus on
        contextual diversity, major themes, and potential gaps. Keep to bullet points.

        Overview JSON:
        {json.dumps(overview, indent=2)}

        Please provide:
        1. A summary of the most common issues
        2. Patterns in the error reasons
        3. Recommendations for improvement
        4. Frequency analysis of each issue type

        Format your response in a clear, structured manner with bullet points and categories.
        """
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You analyze thematic clusters succinctly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    # =============================================================
    # Embeddings & Clustering
    # =============================================================
    def _create_embeddings(self, processed_entries: List[Dict[str, Any]]):
        texts = [e["context_summary"] or e["combined_summary"] for e in processed_entries]
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = model.encode(texts, show_progress_bar=False)
        return X, texts

    def _subcluster(self, X, processed_entries, num_clusters, subcluster_k=None, clustering_method="kmeans"):
        # Cluster the samples
        n_entries = len(processed_entries)
        
        if clustering_method.lower() == "hdbscan":
            entry_labels, clustering_quality = self._subcluster_hdbscan(X, processed_entries, num_clusters)
        else:  # Default to kmeans
            if not subcluster_k:
                subcluster_k = max(2, min(max(int(math.sqrt(n_entries)), num_clusters * 2), n_entries))
            subcluster_k = min(subcluster_k, n_entries)
            clusterer = KMeans(n_clusters=subcluster_k, n_init=10, random_state=42)
            entry_labels = clusterer.fit_predict(X)
            
            # Calculate clustering quality metrics
            clustering_quality = self._calculate_clustering_quality(X, numeric_labels=entry_labels)
        
        # Reduce to 2 dimensions
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n_entries // 3)))
        coords_2d = tsne.fit_transform(X)

        # Subcluster labeling: sample up to 3 contexts per subcluster, use LLM label
        subcluster_samples: Dict[int, List[str]] = {}
        texts = [e["context_summary"] or e["combined_summary"] for e in processed_entries]
        for sc_id in set(entry_labels):
            if sc_id == -1:  # Skip noise points for HDBSCAN
                continue
            member_indices = [i for i, lab in enumerate(entry_labels) if lab == sc_id]
            if not member_indices:
                continue
            sample_indices = random.sample(member_indices, min(3, len(member_indices)))
            samples = [texts[i] for i in sample_indices]
            subcluster_samples[sc_id] = samples
        
        # Assign LLM label to each subcluster and to entries
        numeric_to_label: Dict[int, str] = {}
        for sc_id in set(entry_labels):
            if sc_id == -1:  # Handle noise points for HDBSCAN
                numeric_to_label[sc_id] = ("misc", "Noise points that don't fit well into any cluster")
            else:
                numeric_to_label[sc_id] = self._llm_label_from_strings(subcluster_samples.get(sc_id, []), label_type="subcluster")
        
        for e, lab_num, coord in zip(processed_entries, entry_labels, coords_2d):
            label_info = numeric_to_label.get(lab_num, ["misc", "Miscellaneous entries"])
            e["subcluster_label"] = label_info[0]
            e["subcluster_description"] = label_info[1]
            e["coordinates"] = [float(coord[0]), float(coord[1])]
        subclusters = self._aggregate_subclusters(processed_entries)

        # drop subcluster description from processed+entries
        for e in processed_entries:
            e.pop("subcluster_description", None)

        # Subcluster-level coordinate averaging
        id_to_entry = {e["id"]: e for e in processed_entries}
        for label, info in subclusters.items():
            coords = [id_to_entry[i]["coordinates"] for i in info["entry_ids"] if i in id_to_entry]
            if coords:
                avg_x = sum(c[0] for c in coords) / len(coords)
                avg_y = sum(c[1] for c in coords) / len(coords)
                info["coordinates"] = [avg_x, avg_y]
            else:
                info["coordinates"] = [0.0, 0.0]
        return subclusters, clustering_quality

    def _subcluster_hdbscan(self, X, processed_entries, num_clusters):
        """Perform HDBSCAN clustering for subclustering."""
        if hdbscan is None:
            raise ImportError("HDBSCAN library is not installed. Please install it with: pip install hdbscan")
        
        n_entries = len(processed_entries)
        
        # HDBSCAN parameters - automatically determine clusters
        min_cluster_size = max(2, min(5, n_entries // 10))  # At least 2, but scale with data size
        min_samples = max(1, min_cluster_size - 1)
        
        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass
            prediction_data=True
        )
        
        # Fit and predict
        entry_labels = clusterer.fit_predict(X)
        
        # Calculate clustering quality metrics
        # Filter out noise points (-1 labels) for quality metrics
        valid_indices = [i for i, label in enumerate(entry_labels) if label != -1]
        if len(valid_indices) > 1 and len(set(entry_labels[i] for i in valid_indices)) > 1:
            valid_X = X[valid_indices]
            valid_labels = [entry_labels[i] for i in valid_indices]
            clustering_quality = self._calculate_clustering_quality(valid_X, numeric_labels=valid_labels)
        else:
            clustering_quality = {
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None,
                "inertia_score": None,
                "cluster_balance": None,
                "quality_summary": "HDBSCAN found insufficient valid clusters for quality assessment"
            }
        
        # Add HDBSCAN-specific metrics
        clustering_quality["noise_points"] = int(sum(1 for label in entry_labels if label == -1))
        clustering_quality["noise_ratio"] = float(clustering_quality["noise_points"] / len(entry_labels))
        clustering_quality["cluster_persistence"] = None
        
        # Add cluster persistence scores if available
        if hasattr(clusterer, 'cluster_persistence_'):
            persistence_scores = clusterer.cluster_persistence_
            if len(persistence_scores) > 0:
                clustering_quality["cluster_persistence"] = float(np.mean(persistence_scores))
        
        return entry_labels, clustering_quality

    def _cluster(self, X, processed_entries, subclusters, num_clusters, clustering_method: str = "kmeans"):
        """Top-level clustering of subcluster centroids.

        If clustering_method == 'hdbscan', use HDBSCAN; otherwise fall back to KMeans.
        Re-uses LLM labeling for human-readable cluster names. Noise (label -1) from
        HDBSCAN is grouped into a 'noise' cluster if present.
        """
        # Build subcluster centroids
        sc_vectors: List[np.ndarray] = []
        sc_labels_ordered: List[str] = []
        for sc_label, info in subclusters.items():
            member_indices = [i for i, e in enumerate(processed_entries) if e["id"] in info["entry_ids"]]
            if not member_indices:
                continue
            sc_vec = np.array([X[i] for i in member_indices]).mean(axis=0)
            sc_vectors.append(sc_vec)
            sc_labels_ordered.append(sc_label)

        # Edge case: no or single subcluster
        if len(sc_vectors) <= 1:
            clusters: Dict[str, Any] = {}
            if sc_labels_ordered:
                only = sc_labels_ordered[0]
                label, desc = self._llm_label_from_strings([only], label_type="cluster")
                clusters[label] = {
                    "weight": subclusters[only]["count"],
                    "subcluster_labels": [only],
                    "subcluster_counts": {only: subclusters[only]["count"]},
                    "description": desc,
                }
            return clusters

        dense = np.vstack(sc_vectors)

        use_hdbscan = clustering_method.lower() == "hdbscan" and hdbscan is not None and len(sc_vectors) >= 2
        clusters: Dict[str, Any] = {}

        if use_hdbscan:
            # Parameter heuristic for small number of subclusters
            min_cluster_size = max(2, min(5, len(sc_vectors)//2 or 2))
            min_samples = min_cluster_size - 1 if min_cluster_size > 2 else 1
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            top_labels = hdb.fit_predict(dense)
            cluster_map: Dict[int, List[str]] = defaultdict(list)
            for lbl, sc_lab in zip(top_labels, sc_labels_ordered):
                cluster_map[lbl].append(sc_lab)

            noise_members = cluster_map.pop(-1, []) if -1 in cluster_map else []

            # Build clusters for real labels
            for cid, members in cluster_map.items():
                subcluster_descriptions = [subclusters[m].get("description", m) for m in members]
                cluster_label, cluster_description = self._llm_label_from_strings(subcluster_descriptions, label_type="cluster")
                base = cluster_label
                suffix = 1
                while cluster_label in clusters:
                    cluster_label = f"{base}_{suffix}"
                    suffix += 1
                weight = sum(subclusters[m]["count"] for m in members)
                clusters[cluster_label] = {
                    "weight": weight,
                    "subcluster_labels": members,
                    "subcluster_counts": {m: subclusters[m]["count"] for m in members},
                    "description": cluster_description,
                }

            # Optional noise cluster
            if noise_members:
                weight = sum(subclusters[m]["count"] for m in noise_members)
                clusters["noise"] = {
                    "weight": weight,
                    "subcluster_labels": noise_members,
                    "subcluster_counts": {m: subclusters[m]["count"] for m in noise_members},
                    "description": "Heterogeneous / low-cohesion subclusters (noise)",
                }
        else:
            # KMeans path (existing behavior)
            if len(sc_vectors) <= num_clusters:
                for lab in sc_labels_ordered:
                    label, desc = self._llm_label_from_strings([lab], label_type="cluster")
                    clusters[label] = {
                        "weight": subclusters[lab]["count"],
                        "subcluster_labels": [lab],
                        "subcluster_counts": {lab: subclusters[lab]["count"]},
                        "description": desc,
                    }
            else:
                kmeans_sc = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
                top_labels = kmeans_sc.fit_predict(dense)
                cluster_map: Dict[int, List[str]] = defaultdict(list)
                for lbl, sc_lab in zip(top_labels, sc_labels_ordered):
                    cluster_map[lbl].append(sc_lab)
                for cid, members in cluster_map.items():
                    subcluster_descriptions = [subclusters[m].get("description", m) for m in members]
                    cluster_label, cluster_description = self._llm_label_from_strings(subcluster_descriptions, label_type="cluster")
                    base = cluster_label
                    suffix = 1
                    while cluster_label in clusters:
                        cluster_label = f"{base}_{suffix}"
                        suffix += 1
                    weight = sum(subclusters[m]["count"] for m in members)
                    clusters[cluster_label] = {
                        "weight": weight,
                        "subcluster_labels": members,
                        "subcluster_counts": {m: subclusters[m]["count"] for m in members},
                        "description": cluster_description,
                    }

        # Coordinate averaging (weighted by subcluster counts)
        for cname, cinfo in clusters.items():
            members = cinfo.get("subcluster_labels", [])
            total_weight = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for m in members:
                sc = subclusters.get(m)
                if not sc:
                    continue
                w = float(sc.get("count", 0))
                coord = sc.get("coordinates", [0.0, 0.0])
                sum_x += coord[0] * w
                sum_y += coord[1] * w
                total_weight += w
            if total_weight > 0:
                cinfo["coordinates"] = [sum_x / total_weight, sum_y / total_weight]
            else:
                cinfo["coordinates"] = [0.0, 0.0]
        return clusters

    # =============================================================
    # Quality Metrics & Interpretation
    # =============================================================
    def _calculate_clustering_quality(self, X, numeric_labels) -> Dict[str, Any]:
        """Calculate various clustering quality metrics.
        
        :param X: Feature matrix (embeddings)
        :param numeric_labels: List of numeric cluster labels, same size as X
        :return: Dictionary with clustering quality metrics
        """
        if len(numeric_labels) < 2:
            return {
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None,
                "inertia_score": None,
                "cluster_balance": None,
                "quality_summary": "Insufficient data for quality assessment"
            }
        

        unique_labels = list(set(numeric_labels))


        quality_metrics = {}
        
        try:
            # Silhouette Score (-1 to 1, higher is better)
            if len(unique_labels) > 1 and len(unique_labels) < len(X):
                quality_metrics["silhouette_score"] = float(silhouette_score(X, numeric_labels))
            else:
                quality_metrics["silhouette_score"] = None
                
            # Calinski-Harabasz Score (higher is better)
            if len(unique_labels) > 1:
                quality_metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, numeric_labels))
            else:
                quality_metrics["calinski_harabasz_score"] = None
                
            # Davies-Bouldin Score (lower is better)
            if len(unique_labels) > 1:
                quality_metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, numeric_labels))
            else:
                quality_metrics["davies_bouldin_score"] = None
                
        except Exception as e:
            # Fallback if sklearn metrics fail
            quality_metrics["silhouette_score"] = None
            quality_metrics["calinski_harabasz_score"] = None
            quality_metrics["davies_bouldin_score"] = None
            
        # Calculate inertia (within-cluster sum of squares)
        try:
            kmeans = KMeans(n_clusters=len(unique_labels), n_init=10, random_state=42)
            kmeans.fit(X)
            quality_metrics["inertia_score"] = float(kmeans.inertia_)
        except Exception:
            quality_metrics["inertia_score"] = None
            
        # Cluster balance (measure of how evenly distributed clusters are)
        cluster_sizes = _Counter(numeric_labels).values()
        total_entries = sum(cluster_sizes)
        if total_entries > 0:
            # Calculate coefficient of variation (lower is more balanced)
            mean_size = total_entries / len(cluster_sizes)
            variance = sum((size - mean_size) ** 2 for size in cluster_sizes) / len(cluster_sizes)
            std_dev = variance ** 0.5
            quality_metrics["cluster_balance"] = 1.0 - (std_dev / mean_size) if mean_size > 0 else 0.0
        else:
            quality_metrics["cluster_balance"] = None
            
        # Generate quality summary
        quality_summary = self._generate_quality_summary(quality_metrics, len(unique_labels), len(X))
        quality_metrics["quality_summary"] = quality_summary
        
        return quality_metrics
    
    def _generate_quality_summary(self, metrics: Dict[str, Any], num_clusters: int, num_entries: int) -> str:
        """Generate a human-readable summary of clustering quality."""
        summary_parts = []
        
        # Silhouette score interpretation
        sil_score = metrics.get("silhouette_score")
        if sil_score is not None:
            if sil_score > 0.7:
                summary_parts.append("excellent cluster separation")
            elif sil_score > 0.5:
                summary_parts.append("good cluster separation")
            elif sil_score > 0.25:
                summary_parts.append("moderate cluster separation")
            else:
                summary_parts.append("poor cluster separation")
        
        # Cluster balance interpretation
        balance = metrics.get("cluster_balance")
        if balance is not None:
            if balance > 0.8:
                summary_parts.append("well-balanced cluster sizes")
            elif balance > 0.6:
                summary_parts.append("reasonably balanced cluster sizes")
            else:
                summary_parts.append("unbalanced cluster sizes")
        
        # Davies-Bouldin score interpretation (lower is better)
        db_score = metrics.get("davies_bouldin_score")
        if db_score is not None:
            if db_score < 0.5:
                summary_parts.append("compact and well-separated clusters")
            elif db_score < 1.0:
                summary_parts.append("moderately compact clusters")
            else:
                summary_parts.append("overlapping clusters")
        
        # HDBSCAN-specific metrics
        noise_ratio = metrics.get("noise_ratio")
        if noise_ratio is not None:
            if noise_ratio < 0.05:
                summary_parts.append("minimal noise points")
            elif noise_ratio < 0.15:
                summary_parts.append("low noise points")
            elif noise_ratio < 0.3:
                summary_parts.append("moderate noise points")
            else:
                summary_parts.append("high noise ratio")
        
        cluster_persistence = metrics.get("cluster_persistence")
        if cluster_persistence is not None:
            if cluster_persistence > 0.8:
                summary_parts.append("highly persistent clusters")
            elif cluster_persistence > 0.5:
                summary_parts.append("moderately persistent clusters")
            else:
                summary_parts.append("low cluster persistence")
        
        if not summary_parts:
            return f"Clustering created {num_clusters} clusters from {num_entries} entries"
        
        return f"Clustering shows {', '.join(summary_parts)} ({num_clusters} clusters from {num_entries} entries)"

    # =============================================================
    # Text Normalization / Summarization
    # =============================================================
    def _normalize(self, obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, list):  # list of messages or strings
            parts = []
            for item in obj:
                if isinstance(item, dict):
                    # Common message schemas: {role, content:[{type:'text', text:...}]}
                    role = item.get("role")
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                txt = c.get("text", "")
                                parts.append(f"{role}: {txt}" if role else txt)
                    elif isinstance(content, str):
                        parts.append(f"{role}: {content}" if role else content)
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if isinstance(obj, dict):
            # Extract textual fields heuristically
            textual = []
            for k, v in obj.items():
                if isinstance(v, (str, int, float)):
                    textual.append(f"{k}: {v}")
            return " | ".join(textual) or json.dumps(obj)[:1000]
        return str(obj)

    def _summarize_text(self, obj: Any, focus: str) -> str:
        text = self._normalize(obj)
        if not text:
            return ""

        prompt = (
            "You will summarize the provided text. Focus ONLY on high-level context (purpose, domain, key entities) "
            "and ignore chit-chat or pleasantries if present. Return a concise paragraph (<=60 words).\n\n"
            f"TEXT TYPE: {focus.upper()}\n---\n{text}\n---\nConcise summary:"
        )
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert summarizer focusing on essential domain context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return text[:250] + ("..." if len(text) > 250 else "")

    def _combine_summaries(self, context_summary: str, conversation_summary: Optional[str]) -> str:
        if conversation_summary:
            return f"Context: {context_summary}\nConversation: {conversation_summary}"
        return context_summary

    # =============================================================
    # Aggregation & Counting
    # =============================================================
    def _load_entries_from_file(self, path: str) -> List[EntryType]:
        """Load entries from a JSON / JSONL file.

        Supports:
        - JSON array of objects
        - A single JSON object (wrapped into a list)
        - Newline-delimited JSON objects (JSONL)
        Returns empty list if file unreadable or format unrecognized.
        """
        with open(path, "r", encoding="utf-8") as f:
            data_str = f.read().strip()
        if not data_str:
            return []
        # Try array / single object first
        try:
            parsed = json.loads(data_str)
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass
        # Fallback: treat as JSONL
        entries: List[EntryType] = []
        for line in data_str.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    entries.append(obj)
            except Exception:
                continue
        return entries
    def _aggregate_subclusters(self, processed_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for e in processed_entries:
            label = e["subcluster_label"]
            mapping.setdefault(label, {"entry_ids": [], "count": 0})
            mapping[label]["entry_ids"].append(e["id"])
            mapping[label]["count"] += 1
            mapping[label]["description"] = e["subcluster_description"]
        return mapping

    def _count_conversation_turns(self, conversation: Any) -> int:
        """Heuristically count conversation turns.

        Supports list of dict messages ({role, content}) or list of strings. For a dict with 'content' list
        (OpenAI style), counts each element that contains text. Fallback: 0 if None, 1 if non-empty string.
        """
        if not conversation:
            return 0
        if isinstance(conversation, str):
            return 1 if conversation.strip() else 0
        if isinstance(conversation, list):
            turns = 0
            for item in conversation:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        # count textual parts
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                                turns += 1
                    elif isinstance(content, str) and content.strip():
                        turns += 1
                    else:
                        # treat dict itself as a turn if it has role and any non-empty textual field
                        if any(isinstance(v, str) and v.strip() for v in item.values()):
                            turns += 1
                elif isinstance(item, str) and item.strip():
                    turns += 1
                else:
                    # unknown item type counts as a turn
                    turns += 1
            return turns
        # Fallback: attempt to derive from dict by counting textual values
        if isinstance(conversation, dict):
            return sum(1 for v in conversation.values() if isinstance(v, str) and v.strip()) or 1
        return 0

    # =============================================================
    # LLM Labeling
    # =============================================================
    def _llm_label_from_strings(self, strings: List[str], label_type: str = "subcluster") -> Tuple[str, str]:
        """
        Given a list of strings (contexts or subcluster labels), return a tuple (label, explanation):
        - label: concise, snake_case label using LLM or fallback
        - explanation: short explanation of the cluster/subcluster meaning
        label_type: 'subcluster' or 'cluster' (for prompt context)
        """
        if not strings:
            return ("misc", "No data available to generate an explanation.")
        label = "misc"
        explanation = ""
        if self.openai_client and self.deployment_name:
            if label_type == "subcluster":
                label_prompt = (
                    "You are to create a concise, 3-5 word, lowercase, snake_case label that best describes the common context or topic of the following samples. "
                    "Focus on the main theme or subject. Return ONLY the label string.\n\n"
                    + "\n---\n".join(strings)
                    + "\n---\nLabel:"
                )
                explanation_prompt = (
                    "Given the following samples, provide a 1-2 sentence explanation of the main theme or subject that unites them. "
                    "Be concise and clear.\n\n"
                    + "\n---\n".join(strings)
                    + "\n---\nExplanation:"
                )
            else:
                
                label_prompt =(
                    "You are to create a concise, 3-5 word, lowercase, snake_case label that best describes the common theme or topic of the following subcluster labels. "
                    "Return ONLY the label string.\n\n"
                )
                # we can take an input for specific analysis type, if it is Error Analysis we add the extra instructions.
                label_prompt = label_prompt + ("try to adhere to one of these categories if applicable:   "
                    "Final_Answer_Missing_Information,  Called_Incorrect_Tool, Incorrect_Tool_Call_Formatting, Terminated_Early_Unexpectedly, Hallucinated_Information, Misunderstood_Tool_Info, Repeatedly_Calling_Same_Tool, Action_Plan_Flawed, Miscellaneous"
                    "you can create new labels if needed"
                )
                label_prompt = label_prompt + "\n\n" + ", ".join(strings) + "\nLabel:"
                
                explanation_prompt = (
                    "Given the following subcluster labels, provide a 1-2 sentence explanation of the main theme or subject that unites them. "
                    "Be concise and clear.\n\n"
                    + ", ".join(strings)
                    + "\nExplanation:"
                )
            try:
                # Get label
                response_label = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You create ultra-concise context-oriented labels in snake_case."},
                        {"role": "user", "content": label_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=20,
                )
                label = response_label.choices[0].message.content.strip()
                label = label.replace(" ", "_")
                label = "_".join(label.split("_")[:5]).lower()
                # Get explanation
                response_expl = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You explain the main theme of a group of samples in 1-2 sentences."},
                        {"role": "user", "content": explanation_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=60,
                )
                explanation = response_expl.choices[0].message.content.strip()
                return (label, explanation)
            except Exception:
                pass
        # fallback: use first 5 significant words from all strings for label, and join sample snippets for explanation
        words = []
        for s in strings:
            words.extend([w.lower() for w in re.findall(r"[A-Za-z0-9]+", s) if len(w) > 2])
        label = "_".join(words[:5]) or "misc"
        explanation = (
            "This cluster/subcluster groups entries with common themes such as: " + ", ".join(strings[:2]) + ("..." if len(strings) > 2 else "")
        )
        return (label, explanation)



def visualize_data_analyzer_2d(data_analysis_results, subcluster=True):
    """
    Visualize the 2D embedding/coordinate space from DataAnalyzer output.
    Args:
        data_analysis_results: Output dict from DataAnalyzer.analyze(...)
        subcluster: If True, show subcluster view; else, entry-level view.
    """
    axes = data_analysis_results.get('axes', ['x','y'])
    entries = data_analysis_results['entries']
    subclusters = data_analysis_results['subclusters']
    clusters = data_analysis_results['clusters']
    # Prepare entry points
    rows_e = []
    for e in entries:
        coord = e.get('coordinates', [None, None])
        rows_e.append({
            axes[0]: coord[0],
            axes[1]: coord[1],
            'entry_id': e['id'],
            'subcluster': e['subcluster_label'],
            'conversation_turns': e['metadata'].get('conversation_turns'),
        })
    df_e = pd.DataFrame(rows_e)
    fig = None
    if not df_e.empty:
        fig = px.scatter(
            df_e,
            x=axes[0],
            y=axes[1],
            color='subcluster',
            hover_data=['entry_id','subcluster','conversation_turns'],
            title='Entry Map Colored by Subcluster'
        )
        fig.update_layout(legend_title_text='Subcluster')
        # Add subcluster labels as text at subcluster coordinates
        for label, info in subclusters.items():
            coord = info.get('coordinates', [None, None])
            if coord[0] is not None and coord[1] is not None:
                fig.add_annotation(
                    x=coord[0],
                    y=coord[1],
                    text=label,
                    showarrow=False,
                    font=dict(size=12, color='black', family='Arial'),
                    bgcolor='rgba(255,255,255,0.0)',
                    opacity=0.8
                )
        # Add cluster labels as larger, bold text at cluster coordinates
        for cname, cinfo in clusters.items():
            coord = cinfo.get('coordinates', [None, None])
            if coord[0] is not None and coord[1] is not None:
                fig.add_annotation(
                    x=coord[0],
                    y=coord[1],
                    text=f'<b>{cname}</b>',
                    showarrow=False,
                    font=dict(size=18, color='black', family='Arial',),
                    bgcolor='rgba(255,255,255,0.0)',
                    opacity=0.95
                )
        fig.show()
    else:
        print('No entries to visualize')
__all__ = ["DataAnalyzer"]