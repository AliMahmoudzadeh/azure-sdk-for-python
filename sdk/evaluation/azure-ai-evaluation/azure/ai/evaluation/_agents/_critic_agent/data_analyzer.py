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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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

    # -------- Public API ---------
    def analyze(
        self,
        entries: List[EntryType],
        num_clusters: int = 8,
        classic_subcluster_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full analysis pipeline.

        :param entries: List of entries each having at minimum a 'context' key.
        :param num_clusters: Maximum number of top-level clusters to return.
        :param classic_subcluster_k: Optional override for number of subclusters in classic method. If None an heuristic is used.
        :return: Hierarchical drill-down friendly report.
        """

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
                    "metadata": enriched_metadata,
                }
            )

        # save the processes entries into a json file
        with open("processed_entries.json", "w") as f:
            json.dump(processed_entries, f)

        extra: Dict[str, Any] = {}

        subclusters, clusters, extra = self._classic_cluster(
            processed_entries,
            num_clusters=num_clusters,
            subcluster_k=classic_subcluster_k,
        )
        llm_analysis = self._analyze_overall(processed_entries, subclusters, clusters)


        axis_names = ["embeddings1", "embeddings2"]

        report = {
            "summary": {
                "total_entries": len(entries),
                "unique_subcluster_labels": len(subclusters),
                "total_clusters": len(clusters),
                "clustering_method": "kmeans",
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
    def _create_embeddings(self, processed_entries: List[Dict[str, Any]]):
        texts = [e["context_summary"] or e["combined_summary"] for e in processed_entries]
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = model.encode(texts, show_progress_bar=False)
        return X, texts

    def _subcluster(self, X, processed_entries, num_clusters, subcluster_k=None):
        n_entries = len(processed_entries)
        if not subcluster_k:
            subcluster_k = max(2, min(max(int(math.sqrt(n_entries)), num_clusters * 2), n_entries))
        subcluster_k = min(subcluster_k, n_entries)
        clusterer = KMeans(n_clusters=subcluster_k, n_init=10, random_state=42)
        entry_labels = clusterer.fit_predict(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n_entries // 3)))
        coords_2d = tsne.fit_transform(X)
        # Subcluster labeling: sample up to 3 contexts per subcluster, use LLM label
        subcluster_samples: Dict[int, List[str]] = {}
        texts = [e["context_summary"] or e["combined_summary"] for e in processed_entries]
        for sc_id in set(entry_labels):
            member_indices = [i for i, lab in enumerate(entry_labels) if lab == sc_id]
            if not member_indices:
                continue
            sample_indices = random.sample(member_indices, min(3, len(member_indices)))
            samples = [texts[i] for i in sample_indices]
            subcluster_samples[sc_id] = samples
        # Assign LLM label to each subcluster and to entries
        numeric_to_label: Dict[int, str] = {}
        for sc_id in set(entry_labels):
            numeric_to_label[sc_id] = self._llm_label_from_strings(subcluster_samples.get(sc_id, []), label_type="subcluster")
        for e, lab_num, coord in zip(processed_entries, entry_labels, coords_2d):
            e["subcluster_label"] = numeric_to_label.get(lab_num, "misc")
            e["coordinates"] = [float(coord[0]), float(coord[1])]
        subclusters = self._aggregate_subclusters(processed_entries)
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
        return subclusters

    def _cluster(self, X, processed_entries, subclusters, num_clusters):
        # Build second-level clusters by clustering subcluster centroids
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
                clusters[self._llm_label_from_strings([lab], label_type="cluster")] = {
                    "weight": subclusters[lab]["count"],
                    "subcluster_labels": [lab],
                    "subcluster_counts": {lab: subclusters[lab]["count"]},
                    "description": lab.replace("_", " "),
                }
        else:
            dense = np.vstack(sc_vectors)
            kmeans_sc = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            top_labels = kmeans_sc.fit_predict(dense)
            cluster_map: Dict[int, List[str]] = defaultdict(list)
            for lbl, sc_lab in zip(top_labels, sc_labels_ordered):
                cluster_map[lbl].append(sc_lab)
            clusters = {}
            for idx, (cid, members) in enumerate(cluster_map.items(), start=1):
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
                    "description": f"Classic clustering group (kmeans)",
                }
        # Cluster-level coordinate averaging
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

    def _classic_cluster(
        self,
        processed_entries: List[Dict[str, Any]],
        num_clusters: int,
        subcluster_k: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        X, _ = self._create_embeddings(processed_entries)
        subclusters = self._subcluster(X, processed_entries, num_clusters, subcluster_k=subcluster_k)
        clusters = self._cluster(X, processed_entries, subclusters, num_clusters)
        extra = {
            "method": "classic",
            "subcluster_k": subcluster_k,
            "embedding": "sbert",
            "clustering_algorithm": "kmeans",
            "dimensionality_reduction": "tsne",
            "warnings": [],
        }
        return subclusters, clusters, extra

    # -------- Coordinate utilities ---------

    # Subcluster and cluster coordinate averaging now handled in _subcluster and _cluster methods





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