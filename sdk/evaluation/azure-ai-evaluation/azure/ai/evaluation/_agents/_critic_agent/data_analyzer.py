"""General purpose DataAnalyzer for clustering conversational/contextual entries.

The analyzer takes a list of entries where each entry contains:
    - context: str | dict | list (background info, task instructions, etc.)
    - conversation: str | dict | list (dialogue turns) – optional
    - metadata: dict (arbitrary attributes) – optional

Processing steps:
1. Summarize (textual) each entry focusing first on context, then (optionally) conversation.
2. Generate a 5-word (<=5 tokens separated by spaces) subcluster label emphasizing CONTEXT meaning.
3. Cluster subcluster labels into up to N higher-level clusters via LLM (or heuristic fallback).
4. Record conversation turn count in metadata as 'conversation_turns'.

Output structure (drill-down friendly):
{
  'summary': { 'total_entries': int, 'unique_subcluster_labels': int, 'total_clusters': int },
  'entries': [
      {
        'id': <id or index>,
        'context_summary': str,
        'conversation_summary': str | None,
        'combined_summary': str,
        'subcluster_label': str,
        'metadata': dict
      }, ...
  ],
  'subclusters': { '<subcluster_label>': {'entry_ids': [...], 'count': int } },
  'clusters': {
       '<cluster_name>': {
            'weight': int,
            'subcluster_labels': [...],
            'subcluster_counts': { label: count },
            'description': str
       }, ...
  },
  'llm_analysis': str | None,
  'raw': <original input list>
}

This mirrors the ErrorAnalyzer pattern but is domain-agnostic.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import os
import json
from collections import defaultdict, Counter as _Counter
import random
import re
import math
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
import hdbscan
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

    # -------- Public API ---------
    def analyze(
        self,
        entries: List[EntryType],
        num_clusters: int = 8,
        use_llm_analysis: bool = True,
        clustering_method: str = "llm",
        classic_subcluster_k: Optional[int] = None,
        classic_cluster_algo: str = "kmeans",  # "kmeans", "agglomerative", "hdbscan", "spectral"
    ) -> Dict[str, Any]:
        """Run full analysis pipeline.

        :param entries: List of entries each having at minimum a 'context' key.
        :param num_clusters: Maximum number of top-level clusters to return.
        :param use_llm_analysis: Whether to generate a natural language report of patterns.
        :param clustering_method: 'llm' (default) to derive subclusters via per-entry labels + optional LLM top grouping,
                                  or 'classic' to use TF-IDF + KMeans (no LLM required) to form subclusters and clusters.
        :param classic_subcluster_k: Optional override for number of subclusters in classic method. If None an heuristic is used.
        :return: Hierarchical drill-down friendly report.
        """

        clustering_method = clustering_method.lower()
        if clustering_method not in {"llm", "classic"}:
            raise ValueError("clustering_method must be 'llm' or 'classic'")

        processed_entries: List[Dict[str, Any]] = []

        # First pass: summarize each entry (shared across methods)
        for idx, entry in enumerate(entries):
            context = entry.get("context")
            conversation = entry.get("conversation")
            metadata = entry.get("metadata", {})
            entry_id = entry.get("id", idx)

            context_summary = str(context) #self._summarize_text(context, focus="context")
            conversation_summary = self._summarize_text(conversation, focus="conversation") if conversation else None
            combined_summary = self._combine_summaries(context_summary, conversation_summary)

            # Placeholder label; will be replaced for classic method
            subcluster_label = (
                self._generate_subcluster_label(context_summary, conversation_summary)
                if clustering_method == "llm"
                else "__unassigned__"
            )

            # Augment metadata with conversation turn count
            turns = self._count_conversation_turns(conversation)
            enriched_metadata = dict(metadata)
            enriched_metadata.setdefault("conversation_turns", turns)

            processed_entries.append(
                {
                    "id": entry_id,
                    "context_summary": context_summary,
                    "conversation_summary": conversation_summary,
                    "combined_summary": combined_summary,
                    "subcluster_label": subcluster_label,
                    "metadata": enriched_metadata,
                }
            )

        # save the processes entries into a json file
        with open("processed_entries.json", "w") as f:
            json.dump(processed_entries, f)

        extra: Dict[str, Any] = {}


        if clustering_method == "classic":
            subclusters, clusters, extra = self._classic_cluster(
                processed_entries,
                num_clusters=num_clusters,
                subcluster_k=classic_subcluster_k,
                cluster_algo=classic_cluster_algo,
            )
        else:  # LLM / heuristic path
            subclusters = self._aggregate_subclusters(processed_entries)
            clusters = self._cluster_subclusters(list(subclusters.keys()), subclusters, num_clusters)
        # Ensure 2D coordinates present for entries + propagate to subclusters/clusters
        self._ensure_coordinates(processed_entries, subclusters, clusters, method=clustering_method)

        llm_analysis = self._analyze_overall(processed_entries, subclusters, clusters) if use_llm_analysis else None

        # Axis naming (top-level metadata for downstream visualization)
        if clustering_method == "classic":
            axis_names = ["embeddings1", "embeddings2"]
        else:  # now also embedding-based for llm path
            axis_names = ["embeddings1", "embeddings2"]

        report = {
            "summary": {
                "total_entries": len(entries),
                "unique_subcluster_labels": len(subclusters),
                "total_clusters": len(clusters),
                "clustering_method": clustering_method,
            },
            "entries": processed_entries,
            "subclusters": subclusters,
            "clusters": clusters,
            "llm_analysis": llm_analysis,
            "axes": axis_names,
            "raw": entries,
        }
        if extra:
            report["classic_metadata"] = extra
        return report

    # -------- Internal helpers ---------
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

    def _generate_subcluster_label(self, context_summary: str, conversation_summary: Optional[str]) -> str:
        base = context_summary or conversation_summary or ""
        if not base:
            return "unspecified"
        prompt = f"""
        Create a 5 word (<=5 words, lowercase, snake_case) label capturing the CORE CONTEXTUAL TOPIC OR TASK of the following summaries.
        Prioritize background context over dialogue chit-chat. Do NOT exceed 5 words. Return ONLY the label string.
        If the context explains an evaluation issue, focus on the issue.
        Summaries:\nContext: {context_summary}\n
        """
        # Conversation: {conversation_summary or '[none]'}

        response = self.openai_client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You create ultra-concise 5-word context-oriented labels in snake_case."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=20,
        )
        label = response.choices[0].message.content.strip()
        # sanitize
        label = label.replace(" ", "_")
        return "_".join(label.split("_")[:5]).lower()


    def _aggregate_subclusters(self, processed_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for e in processed_entries:
            label = e["subcluster_label"]
            mapping.setdefault(label, {"entry_ids": [], "count": 0})
            mapping[label]["entry_ids"].append(e["id"])
            mapping[label]["count"] += 1
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

    def _cluster_subclusters(
        self,
        labels: List[str],
        subclusters: Dict[str, Dict[str, Any]],
        num_clusters: int,
    ) -> Dict[str, Any]:
        if not labels:
            return {}
        if not self.openai_client or not self.deployment_name:
            # heuristic grouping: by first keyword
            groups = defaultdict(list)
            for l in labels:
                key = l.split("_")[0]
                groups[key].append(l)
            clusters: Dict[str, Any] = {}
            for idx, (g, members) in enumerate(groups.items()):
                if idx >= num_clusters:
                    break
                weight = sum(subclusters[m]["count"] for m in members)
                clusters[f"Cluster_{idx+1}_{g}"] = {
                    "weight": weight,
                    "subcluster_labels": members,
                    "subcluster_counts": {m: subclusters[m]["count"] for m in members},
                    "description": f"Heuristic group sharing prefix '{g}'",
                }
            return clusters

        freq = {l: subclusters[l]["count"] for l in labels}
        prompt = f"""
        You are to group the following subcluster labels (with counts) into at most {num_clusters} higher-level clusters.
        Provide meaningful, human-readable cluster names (snake_case) that reflect thematic CONTEXT.
        Every subcluster label must appear exactly once.
        Return STRICT JSON with this schema:
        {{
          "clusters": {{
             "cluster_name": {{
                "subcluster_labels": ["label1", "label2"],
                "description": "short description"
             }}, ...
          }}
        }}

        Subcluster counts:
        {json.dumps(freq, indent=2)}
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You cluster concise labels. Always return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            clusters_json = data.get("clusters", {})
            clusters: Dict[str, Any] = {}
            for name, info in clusters_json.items():
                member_labels = info.get("subcluster_labels", [])
                weight = sum(subclusters.get(m, {}).get("count", 0) for m in member_labels)
                clusters[name] = {
                    "weight": weight,
                    "subcluster_labels": member_labels,
                    "subcluster_counts": {m: subclusters.get(m, {}).get("count", 0) for m in member_labels},
                    "description": info.get("description", "") or "",
                }
            return clusters
        except Exception:
            # fallback simple all-in-one cluster
            total = sum(subclusters[l]["count"] for l in labels)
            return {
                "Cluster_1_all": {
                    "weight": total,
                    "subcluster_labels": labels,
                    "subcluster_counts": {l: subclusters[l]["count"] for l in labels},
                    "description": "Fallback single cluster",
                }
            }

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

    # -------- Classic (embedding-based) clustering ---------
    def _classic_cluster(
        self,
        processed_entries: List[Dict[str, Any]],
        num_clusters: int,
        subcluster_k: Optional[int] = None,
        cluster_algo: str = "kmeans",
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Perform classic clustering using SBERT embeddings and various algorithms.

        Returns (subclusters, clusters, extra_metadata)
        """
        texts = [e["context_summary"] or e["combined_summary"] for e in processed_entries]
        n_entries = len(processed_entries)
        if not subcluster_k:
            subcluster_k = max(2, min(max(int(math.sqrt(n_entries)), num_clusters * 2), n_entries))
        subcluster_k = min(subcluster_k, n_entries)

        # SBERT embedding
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = model.encode(texts, show_progress_bar=False)

        # Clustering
        cluster_algo = (cluster_algo or "kmeans").lower()
        if cluster_algo == "kmeans":
            clusterer = KMeans(n_clusters=subcluster_k, n_init=10, random_state=42)
            entry_labels = clusterer.fit_predict(X)
        elif cluster_algo == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=subcluster_k)
            entry_labels = clusterer.fit_predict(X)
        elif cluster_algo == "spectral":
            clusterer = SpectralClustering(n_clusters=subcluster_k, affinity="nearest_neighbors", random_state=42)
            entry_labels = clusterer.fit_predict(X)
        elif cluster_algo == "hdbscan":
            if hdbscan is None:
                raise ImportError("hdbscan is not installed")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_entries // (subcluster_k or 2)))
            entry_labels = clusterer.fit_predict(X)
            # hdbscan can assign -1 for noise, remap to unique clusters
            unique_labels = sorted(set(entry_labels) - {-1})
            label_map = {l: i for i, l in enumerate(unique_labels)}
            entry_labels = [label_map.get(l, -1) for l in entry_labels]
        else:
            raise ValueError(f"Unknown clustering algorithm: {cluster_algo}")

        # 2D reduction with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n_entries // 3)))
        coords_2d = tsne.fit_transform(X)

        # Subcluster labeling: sample up to 3 contexts per subcluster, use LLM to generate a tag
        subcluster_samples: Dict[int, List[str]] = {}
        for sc_id in set(entry_labels):
            member_indices = [i for i, lab in enumerate(entry_labels) if lab == sc_id]
            if not member_indices:
                continue
            # Randomly sample up to 3 contexts from the subcluster
            sample_indices = random.sample(member_indices, min(3, len(member_indices)))
            samples = [texts[i] for i in sample_indices]
            subcluster_samples[sc_id] = samples


        # Map numeric subcluster to textual label (ensure uniqueness)
        used_labels = set()
        numeric_to_label: Dict[int, str] = {}
        for sc_id in set(entry_labels):
            cand = self._llm_label_from_strings(subcluster_samples.get(sc_id, []), label_type="subcluster")
            base = cand
            suffix = 1
            while cand in used_labels:
                cand = f"{base}_{suffix}"
                suffix += 1
            used_labels.add(cand)
            numeric_to_label[sc_id] = cand

        # Assign back to entries
        for e, lab_num, coord in zip(processed_entries, entry_labels, coords_2d):
            e["subcluster_label"] = numeric_to_label.get(lab_num, "misc")
            e["embedding_2d"] = [float(coord[0]), float(coord[1])]

        subclusters = self._aggregate_subclusters(processed_entries)

        # Build second-level clusters by clustering subcluster centroids
        # Represent each subcluster by mean vector of its members
        sc_vectors = []
        sc_labels_ordered = []
        for sc_label, info in subclusters.items():
            member_indices = [i for i, e in enumerate(processed_entries) if e["id"] in info["entry_ids"]]
            if not member_indices:
                continue
            sc_vec = np.array([X[i] for i in member_indices]).mean(axis=0)
            sc_vectors.append(sc_vec)
            sc_labels_ordered.append(sc_label)

        if len(sc_vectors) <= num_clusters:
            clusters: Dict[str, Any] = {}
            for idx, lab in enumerate(sc_labels_ordered):
                clusters[f"Cluster_{idx+1}"] = {
                    "weight": subclusters[lab]["count"],
                    "subcluster_labels": [lab],
                    "subcluster_counts": {lab: subclusters[lab]["count"]},
                    "description": lab.replace("_", " "),
                }
        else:
            dense = np.vstack(sc_vectors)
            # Use KMeans for top-level clusters for simplicity
            kmeans_sc = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            top_labels = kmeans_sc.fit_predict(dense)
            cluster_map: Dict[int, List[str]] = defaultdict(list)
            for lbl, sc_lab in zip(top_labels, sc_labels_ordered):
                cluster_map[lbl].append(sc_lab)
            # Use LLM to name clusters based on their subcluster labels
            clusters = {}
            for idx, (cid, members) in enumerate(cluster_map.items(), start=1):
                # Use the merged label method for cluster naming
                cluster_label = self._llm_label_from_strings(members, label_type="cluster")
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
                    "description": f"Classic clustering group ({cluster_algo})",
                }

        extra = {
            "method": "classic",
            "subcluster_k": subcluster_k,
            "embedding": "sbert",
            "clustering_algorithm": cluster_algo,
            "dimensionality_reduction": "tsne",
            "warnings": [],
        }
        return subclusters, clusters, extra

    # -------- Coordinate utilities ---------
    def _ensure_coordinates(
        self,
        processed_entries: List[Dict[str, Any]],
        subclusters: Dict[str, Dict[str, Any]],
        clusters: Dict[str, Any],
        method: str,
    ) -> None:
        """Populate per-entry coordinates (random for llm, projection for classic) and aggregate averages.

        Adds 'coordinates': [x, y] to each entry, each subcluster, and each cluster.
        Cluster coordinates are weighted averages (by subcluster counts).
        Subcluster coordinates are averages over member entry coordinates.
        """
        # 1. Entry-level
        # For classic: embedding_2d already present. For llm: build TF-IDF + SVD projection similar to classic.
        precomputed_coords = None
        if method == "llm":
            try:
                texts = [e.get("context_summary") or e.get("combined_summary") or "" for e in processed_entries]
                if TfidfVectorizer is not None and TruncatedSVD is not None:
                    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
                    X = vectorizer.fit_transform(texts)
                    reducer = TruncatedSVD(n_components=2, random_state=42)
                    precomputed_coords = reducer.fit_transform(X)
                else:
                    precomputed_coords = None
            except Exception:
                precomputed_coords = None
        for idx, e in enumerate(processed_entries):
            if "coordinates" in e:
                continue
            if method == "classic" and "embedding_2d" in e:
                e["coordinates"] = e["embedding_2d"]
            elif method == "llm" and precomputed_coords is not None:
                e["coordinates"] = [float(precomputed_coords[idx][0]), float(precomputed_coords[idx][1])]
            else:  # fallback random
                e["coordinates"] = [random.uniform(-1, 1), random.uniform(-1, 1)]

        # Build quick lookup by entry id
        id_to_entry = {e["id"]: e for e in processed_entries}

        # 2. Subcluster-level averages
        for label, info in subclusters.items():
            coords = [id_to_entry[i]["coordinates"] for i in info["entry_ids"] if i in id_to_entry]
            if coords:
                avg_x = sum(c[0] for c in coords) / len(coords)
                avg_y = sum(c[1] for c in coords) / len(coords)
                info["coordinates"] = [avg_x, avg_y]
            else:
                info["coordinates"] = [0.0, 0.0]

        # 3. Cluster-level weighted averages (by subcluster count)
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





    def _llm_label_from_strings(self, strings: List[str], label_type: str = "subcluster") -> str:
        """
        Given a list of strings (contexts or subcluster labels), return a concise, snake_case label using LLM or fallback.
        label_type: 'subcluster' or 'cluster' (for prompt context)
        """
        if not strings:
            return "misc"
        if self.openai_client and self.deployment_name:
            if label_type == "subcluster":
                prompt = (
                    "You are to create a concise, 3-5 word, lowercase, snake_case label that best describes the common context or topic of the following samples. "
                    "Focus on the main theme or subject. Return ONLY the label string.\n\n"
                    + "\n---\n".join(strings)
                    + "\n---\nLabel:"
                )
            else:
                prompt = (
                    "You are to create a concise, 3-5 word, lowercase, snake_case label that best describes the common theme or topic of the following subcluster labels. "
                    "Return ONLY the label string.\n\n"
                    + ", ".join(strings)
                    + "\nLabel:"
                )
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You create ultra-concise context-oriented labels in snake_case."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=20,
                )
                label = response.choices[0].message.content.strip()
                label = label.replace(" ", "_")
                return "_".join(label.split("_")[:5]).lower()
            except Exception:
                pass
        # fallback: use first 5 significant words from all strings
        words = []
        for s in strings:
            words.extend([w.lower() for w in re.findall(r"[A-Za-z0-9]+", s) if len(w) > 2])
        
        return "_".join(words[:5]) or "misc"
    
__all__ = ["DataAnalyzer"]