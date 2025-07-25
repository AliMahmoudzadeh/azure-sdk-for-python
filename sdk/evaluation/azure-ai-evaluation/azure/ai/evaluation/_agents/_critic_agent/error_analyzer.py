import pandas as pd
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import openai
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud


class ErrorAnalyzer:
    """
    A tool for analyzing evaluation results to identify patterns in non-perfect evaluations.
    """
    
    def __init__(self, openai_client: Optional[AzureOpenAI] = None):
        """
        Initialize the ErrorAnalyzer.
        
        Args:
            openai_client: Optional OpenAI client for LLM analysis. If not provided,
                          will attempt to create one from environment variables.
        """
        # Load environment variables
        load_dotenv()
        
        self.openai_client = openai_client
        if not self.openai_client:
            # Try to initialize from environment variables
            try:
                self.openai_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                # Store deployment name
                self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
                print("using model", self.deployment_name)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                print("LLM analysis will not be available.")
                self.deployment_name = None
        else:
            # If client is provided, use default deployment name
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    def filter_imperfect_evaluations(self, evaluations: List[Dict[str, Any]], 
                                   fails_only: bool = False) -> List[Dict[str, Any]]:
        """
        Filter evaluations to keep only non-perfect ones.
        
        Args:
            evaluations: List of evaluation dictionaries
            fails_only: If True, only keep evaluations where result is 'fail'.
                        If False, keep all evaluations where score is not perfect.
        
        Returns:
            Filtered list of evaluations
        """
        imperfect_evaluations = []
        
        for eval_data in evaluations:
            thread_id = eval_data.get('thread_id')
            results = eval_data.get('results', {})
            
            eval_issues = {}
            
            for eval_name, eval_result in results.items():
                if fails_only:
                    # Only include if result is 'fail'
                    if eval_result.get(f'{eval_name.lower()}_result') == 'fail':
                        eval_issues[eval_name] = eval_result
                else:
                    # Include if score is not perfect (assuming perfect is 5.0)
                    score_key = f'{eval_name.lower()}_score'
                    # Try different possible score key formats
                    score = eval_result.get(score_key) or eval_result.get(eval_name.lower())
                    
                    if score is not None and score < 5.0:
                        eval_issues[eval_name] = eval_result
            
            if eval_issues:
                imperfect_evaluations.append({
                    'thread_id': thread_id,
                    'results': eval_issues
                })
        
        return imperfect_evaluations
    
    def extract_error_patterns(self, imperfect_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract patterns from imperfect evaluations.
        
        Args:
            imperfect_evaluations: List of filtered evaluation results
            
        Returns:
            Dictionary containing error patterns and statistics
        """
        patterns = {
            'evaluation_types': defaultdict(int),
            'score_distribution': defaultdict(lambda: defaultdict(int)),
            'reasons': defaultdict(list),
            'total_issues': 0,
            'total_threads': len(imperfect_evaluations)
        }
        
        for eval_data in imperfect_evaluations:
            for eval_name, eval_result in eval_data['results'].items():
                patterns['evaluation_types'][eval_name] += 1
                patterns['total_issues'] += 1
                
                # Extract score
                score_key = f'{eval_name.lower()}_score'
                score = eval_result.get(score_key) or eval_result.get(eval_name.lower())
                if score is not None:
                    patterns['score_distribution'][eval_name][score] += 1
                
                # Extract reason
                reason_key = f'{eval_name.lower()}_reason'
                reason = eval_result.get(reason_key)
                if reason:
                    patterns['reasons'][eval_name].append(reason)
        
        return patterns
    
    def generate_error_names(self, reasons: List[str], batch_size: int = 10) -> List[str]:
        """
        Generate short error names for a list of reasons using LLM.
        
        Args:
            reasons: List of detailed error reasons
            batch_size: Number of reasons to process in each API call
            
        Returns:
            List of short error names corresponding to each reason
        """
        if not self.openai_client:
            print("Warning: OpenAI client not available, returning generic error names")
            return [f"Error_{i+1}" for i in range(len(reasons))]
        
        error_names = []
        
        # Process reasons in batches to avoid token limits
        for i in range(0, len(reasons), batch_size):
            batch_reasons = reasons[i:i+batch_size]
            
            prompt = f"""
            For each of the following error reasons, generate a short, descriptive error name (2-4 words max).
            The error name should capture the essence of the issue in a concise way.
            
            Format your response as a JSON list with exactly {len(batch_reasons)} items, one for each reason:
            
            Reasons:
            {json.dumps(batch_reasons, indent=2)}
            
            Example format:
            ["unrelated_information_returned", "tool_is_not_used", "wrong_tool_is_called", ...]

            Response (JSON list only):
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at categorizing and naming error patterns. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                # Try to parse JSON response
                try:
                    batch_names = json.loads(content)
                    if isinstance(batch_names, list) and len(batch_names) == len(batch_reasons):
                        error_names.extend(batch_names)
                    else:
                        print(f"Warning: Unexpected response format for batch {i//batch_size + 1}")
                        error_names.extend([f"Error_{i+j+1}" for j in range(len(batch_reasons))])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON response for batch {i//batch_size + 1}")
                    error_names.extend([f"Error_{i+j+1}" for j in range(len(batch_reasons))])
                    
            except Exception as e:
                print(f"Error generating error names for batch {i//batch_size + 1}: {e}")
                error_names.extend([f"Error_{i+j+1}" for j in range(len(batch_reasons))])
        
        return error_names
    
    def cluster_error_names(self, error_names: List[str], num_clusters: int = 10) ->  Dict[str, Any]:
        """
        Cluster error names into groups with weights using LLM.
        
        Args:
            error_names: List of short error names
            num_clusters: Maximum number of clusters to create
            
        Returns:
            Dictionary with cluster information and weights
        """
        if not self.openai_client:
            print("Warning: OpenAI client not available, returning simple frequency clusters")
            name_counts = Counter(error_names)
            clusters = {}
            for i, (name, count) in enumerate(name_counts.most_common(num_clusters)):
                clusters[f"Cluster_{i+1}_{name}"] = {
                    'weight': count,
                    'errors': [name] * count,
                    'description': f"Issues related to {name}"
                }
            return {'clusters': clusters, 'total_errors': len(error_names)}
        
        # Count frequencies
        name_counts = Counter(error_names)
        unique_names = list(name_counts.keys())
        
        prompt = f"""
        You have {len(error_names)} error instances with {len(unique_names)} unique error types.
        Group these error names into {min(num_clusters, len(unique_names))} meaningful clusters.
        
        Error names with their frequencies:
        {json.dumps(dict(name_counts), indent=2)}
        
        Create clusters that group semantically similar errors together.
        Each cluster should have:
        1. A descriptive cluster name
        2. A list of error names that belong to it
        3. A brief description of what the cluster represents
        
        Format your response as JSON:
        {{
            "clusters": {{
                "Cluster_Name_1": {{
                    "error_names": ["Error1", "Error2"],
                    "description": "Brief description of this cluster"
                }},
                "Cluster_Name_2": {{
                    "error_names": ["Error3", "Error4"],
                    "description": "Brief description of this cluster"
                }}
            }}
        }}
        
        Make sure every error name appears in exactly one cluster.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at clustering and categorizing error patterns. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            cluster_data = json.loads(content)
            
            # Calculate weights and build final structure
            final_clusters = {}
            total_assigned = 0
            
            for cluster_name, cluster_info in cluster_data.get('clusters', {}).items():
                cluster_errors = cluster_info.get('error_names', [])
                cluster_weight = sum(name_counts.get(name, 0) for name in cluster_errors)
                total_assigned += cluster_weight
                
                if cluster_weight > 0:
                    final_clusters[cluster_name] = {
                        'weight': cluster_weight,
                        'errors': cluster_errors,
                        'error_counts': {name: name_counts.get(name, 0) for name in cluster_errors},
                        'description': cluster_info.get('description', 'No description provided')
                    }
            
            return {
                'clusters': final_clusters,
                'total_errors': len(error_names),
                'total_assigned': total_assigned,
                'coverage': total_assigned / len(error_names) if error_names else 0
            }
            
        except Exception as e:
            print(f"Error during clustering: {e}")
            # Fallback to simple frequency-based clustering
            clusters = {}
            for i, (name, count) in enumerate(name_counts.most_common(num_clusters)):
                clusters[f"Cluster_{i+1}_{name}"] = {
                    'weight': count,
                    'errors': [name],
                    'error_counts': {name: count},
                    'description': f"Issues related to {name}"
                }
            return {
                'clusters': clusters,
                'total_errors': len(error_names),
                'total_assigned': sum(c['weight'] for c in clusters.values()),
                'coverage': 1.0
            }

    def analyze_with_llm(self, patterns: Dict[str, Any]) -> str:
        """
        Use LLM to analyze error patterns and provide insights.
        
        Args:
            patterns: Error patterns dictionary from extract_error_patterns
            
        Returns:
            LLM analysis summary
        """
        if not self.openai_client:
            return "LLM analysis not available - OpenAI client not initialized"
        
        # Prepare summary data for LLM
        summary_data = {
            'total_issues': patterns['total_issues'],
            'total_threads': patterns['total_threads'],
            'evaluation_types_frequency': dict(patterns['evaluation_types']),
            'score_distributions': {k: dict(v) for k, v in patterns['score_distribution'].items()},
            'sample_reasons': {k: v[:5] for k, v in patterns['reasons'].items()}  # First 5 reasons per type
        }
        
        prompt = f"""
        Analyze the following evaluation error patterns and provide insights:

        {json.dumps(summary_data, indent=2)}

        Please provide:
        1. A summary of the most common issues
        2. Patterns in the error reasons
        3. Recommendations for improvement
        4. Frequency analysis of each issue type

        Format your response in a clear, structured manner with bullet points and categories.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing evaluation data and identifying patterns in AI system performance issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during LLM analysis: {str(e)}"
    
    def generate_enhanced_report(self, evaluations: List[Dict[str, Any]], 
                               fails_only: bool = False,
                               use_llm_analysis: bool = True,
                               num_clusters: int = 10) -> Dict[str, Any]:
        """
        Generate an enhanced error analysis report with error name clustering.
        
        Args:
            evaluations: List of evaluation dictionaries
            fails_only: Filter mode for evaluations
            use_llm_analysis: Whether to include LLM analysis
            num_clusters: Maximum number of error clusters to create
            
        Returns:
            Enhanced analysis report with error clustering
        """
        print("Step 1: Filtering imperfect evaluations...")
        imperfect_evals = self.filter_imperfect_evaluations(evaluations, fails_only)
        
        print("Step 2: Extracting error patterns...")
        patterns = self.extract_error_patterns(imperfect_evals)
        
        print("Step 3: Generating short error names...")
        all_reasons = []
        reason_to_eval_type = {}
        
        for eval_type, reasons in patterns['reasons'].items():
            for reason in reasons:
                all_reasons.append(reason)
                reason_to_eval_type[reason] = eval_type
        
        error_names = self.generate_error_names(all_reasons)

        print("Step 4: Clustering error names...")
        cluster_results = self.cluster_error_names(error_names, num_clusters)
        print("Step 5: Generating LLM analysis...")
        llm_analysis = None
        if use_llm_analysis:
            # Enhanced prompt with cluster information
            enhanced_patterns = patterns.copy()
            enhanced_patterns['error_clusters'] = cluster_results
            llm_analysis = self.analyze_with_llm(enhanced_patterns)

        # Compile enhanced report
        report = {
            'summary': {
                'total_evaluations': len(evaluations),
                'imperfect_evaluations': len(imperfect_evals),
                'total_issues': patterns['total_issues'],
                'filter_mode': 'fails_only' if fails_only else 'non_perfect',
                'unique_error_types': len(set(error_names)),
                'total_reasons_analyzed': len(all_reasons)
            },
            'patterns': patterns,
            'error_names': error_names,
            'error_clusters': cluster_results,
            'reason_mapping': {
                'reasons_to_names': dict(zip(all_reasons, error_names)),
                'reasons_to_eval_types': reason_to_eval_type
            },
            'llm_analysis': llm_analysis,
            'raw_imperfect_data': imperfect_evals
        }
        
        return report

    
    def visualize_cluster_results(self, cluster_results: Dict[str, Any], 
                                figsize: tuple = (15, 12),
                                save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualizations for cluster results.
        
        Args:
            cluster_results: Dictionary containing cluster information from cluster_error_names
            figsize: Figure size for the plots
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")
            print("Please install: pip install matplotlib seaborn")
            return
        
        if not cluster_results or 'clusters' not in cluster_results:
            print("No cluster data available for visualization")
            return
            
        clusters = cluster_results['clusters']
        if not clusters:
            print("No clusters found in results")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Error Cluster Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Cluster Weight Distribution (Bar Chart)
        ax1 = axes[0, 0]
        cluster_names = list(clusters.keys())
        weights = [clusters[name]['weight'] for name in cluster_names]
        
        # Truncate long cluster names for display
        display_names = [name[:30] + '...' if len(name) > 30 else name for name in cluster_names]
        
        bars = ax1.barh(range(len(display_names)), weights, color=sns.color_palette("husl", len(cluster_names)))
        ax1.set_yticks(range(len(display_names)))
        ax1.set_yticklabels(display_names, fontsize=8)
        ax1.set_xlabel('Weight (Number of Errors)')
        ax1.set_title('Cluster Weights Distribution')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax1.text(weight + max(weights) * 0.01, i, f'{weight}', 
                    va='center', ha='left', fontsize=8)
        
        # 2. Cluster Coverage Pie Chart
        ax2 = axes[0, 1]
        
        # Prepare data for pie chart
        pie_data = []
        pie_labels = []
        colors = sns.color_palette("husl", len(cluster_names) + 1)
        
        for i, (name, cluster_info) in enumerate(clusters.items()):
            pie_data.append(cluster_info['weight'])
            # Truncate labels for pie chart
            label = name.split('_')[0] if '_' in name else name
            pie_labels.append(f"{label[:15]}...")
        
        # Add "uncovered" if coverage < 100%
        total_coverage = cluster_results.get('coverage', 1.0)
        if total_coverage < 1.0:
            uncovered = cluster_results.get('total_errors', 0) * (1 - total_coverage)
            pie_data.append(uncovered)
            pie_labels.append('Uncovered')
        
        wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title(f'Error Distribution\n(Coverage: {total_coverage:.1%})')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
        
        # 3. Error Types per Cluster (Stacked Bar)
        ax3 = axes[1, 0]
        
        # Prepare data for stacked bar chart
        cluster_error_counts = {}
        all_error_types = set()
        
        for cluster_name, cluster_info in clusters.items():
            error_counts = cluster_info.get('error_counts', {})
            cluster_error_counts[cluster_name] = error_counts
            all_error_types.update(error_counts.keys())
        
        if all_error_types:
            # Create matrix for stacked bar
            error_types = list(all_error_types)
            cluster_names_short = [name.split('_')[0][:10] for name in cluster_names]
            
            # Create data matrix
            data_matrix = []
            for cluster_name in cluster_names:
                row = []
                for error_type in error_types:
                    count = cluster_error_counts[cluster_name].get(error_type, 0)
                    row.append(count)
                data_matrix.append(row)
            
            # Create stacked bar chart
            data_matrix = np.array(data_matrix).T  # Transpose for proper stacking
            bottom = np.zeros(len(cluster_names_short))
            
            colors_stack = sns.color_palette("Set3", len(error_types))
            
            for i, (error_type, color) in enumerate(zip(error_types, colors_stack)):
                ax3.bar(cluster_names_short, data_matrix[i], bottom=bottom, 
                       label=error_type[:15], color=color, alpha=0.8)
                bottom += data_matrix[i]
            
            ax3.set_xlabel('Clusters')
            ax3.set_ylabel('Number of Errors')
            ax3.set_title('Error Types Distribution by Cluster')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add legend, but limit to avoid overcrowding
            if len(error_types) <= 8:
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                ax3.text(0.02, 0.98, f'Showing {len(error_types)} error types', 
                        transform=ax3.transAxes, fontsize=8, verticalalignment='top')
        else:
            ax3.text(0.5, 0.5, 'No error type data available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Error Types Distribution by Cluster')
        
        # 4. Cluster Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')  # Turn off axis for text display
        
        # Create summary statistics
        stats_text = []
        stats_text.append("CLUSTER ANALYSIS SUMMARY")
        stats_text.append("=" * 25)
        stats_text.append(f"Total Clusters: {len(clusters)}")
        stats_text.append(f"Total Errors: {cluster_results.get('total_errors', 0)}")
        stats_text.append(f"Coverage: {cluster_results.get('coverage', 0):.1%}")
        stats_text.append(f"Assigned Errors: {cluster_results.get('total_assigned', 0)}")
        stats_text.append("")
        stats_text.append("TOP 5 CLUSTERS:")
        stats_text.append("-" * 15)
        
        # Sort clusters by weight and show top 5
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['weight'], reverse=True)
        for i, (name, info) in enumerate(sorted_clusters[:5], 1):
            weight = info['weight']
            percentage = (weight / cluster_results.get('total_errors', 1)) * 100
            cluster_name_short = name.split('_')[0][:20]
            stats_text.append(f"{i}. {cluster_name_short}")
            stats_text.append(f"   Weight: {weight} ({percentage:.1f}%)")
            
            # Add description if available
            description = info.get('description', '')
            if description and len(description) < 50:
                stats_text.append(f"   {description}")
            stats_text.append("")
        
        # Display the text
        full_text = '\n'.join(stats_text)
        ax4.text(0.05, 0.95, full_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        # Create additional word cloud visualization if possible
        self._create_cluster_wordcloud(clusters, figsize=(12, 8))
    
    def _create_cluster_wordcloud(self, clusters: Dict[str, Any], figsize: tuple = (12, 8)) -> None:
        """
        Create a word cloud visualization of error clusters.
        
        Args:
            clusters: Dictionary containing cluster information
            figsize: Figure size for the word cloud
        """
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
        except ImportError:
            print("WordCloud not available. Install with: pip install wordcloud")
            return
        
        # Prepare text data for word cloud
        cluster_text = []
        for cluster_name, cluster_info in clusters.items():
            weight = cluster_info['weight']
            errors = cluster_info.get('errors', [])
            
            # Add cluster name repeated by weight for frequency
            cluster_name_clean = cluster_name.replace('_', ' ')
            cluster_text.extend([cluster_name_clean] * min(weight, 10))  # Cap to avoid domination
            
            # Add error names
            for error in errors[:3]:  # Limit to top 3 errors per cluster
                if isinstance(error, str):
                    cluster_text.append(error.replace('_', ' '))
        
        if not cluster_text:
            print("No text data available for word cloud")
            return
        
        # Create word cloud
        text = ' '.join(cluster_text)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=50,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        # Display word cloud
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Error Clusters Word Cloud', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()