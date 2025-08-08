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
    
    def _camel_to_snake(self, name: str) -> str:
        """
        Convert camelCase to snake_case.
        
        Args:
            name: String in camelCase format
            
        Returns:
            String in snake_case format
        """
        import re
        # Insert underscore before uppercase letters that follow lowercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before uppercase letters that follow lowercase letters or digits
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
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
            conversation = eval_data.get('conversation', {})

            eval_issues = {}
            
            for eval_name, eval_result in results.items():
                eval_field_name = self._camel_to_snake(eval_name)
                if fails_only:
                    # Only include if result is 'fail'
                    if eval_result.get(f'{eval_field_name}_result') == 'fail':
                        eval_issues[eval_name] = eval_result
                else:
                    # Include if score is not perfect (assuming perfect is 5.0)
                    score =  eval_result.get(eval_field_name)
                    
                    if score is not None and score < 5.0:
                        eval_issues[eval_name] = eval_result
            
            if eval_issues:
                imperfect_evaluations.append({
                    'thread_id': thread_id,
                    'results': eval_issues,
                    'conversation': conversation
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
            # 'score_distribution': defaultdict(lambda: defaultdict(int)),
            'reasons': defaultdict(list),
            'total_issues': 0,
            'total_threads': len(imperfect_evaluations)
        }
        
        for eval_data in imperfect_evaluations:
            for eval_name, eval_result in eval_data['results'].items():
                if eval_name.lower() in eval_result:
                    eval_field_name = eval_name.lower()
                else:
                    eval_field_name = self._camel_to_snake(eval_name)
                patterns['evaluation_types'][eval_name] += 1
                patterns['total_issues'] += 1
                
                # Extract score
                score =  eval_result.get(eval_field_name)
                # if score is not None:
                #     patterns['score_distribution'][eval_name][score] += 1
                
                # Extract reason
                reason_key = f'{eval_field_name}_reason'
                reason = eval_result.get(reason_key)
                if reason:
                    patterns['reasons'][eval_name].append(reason)
        
        return patterns
    
    def generate_error_tags(self, reasons: List[str], batch_size: int = 10) -> List[str]:
        """
        Generate short error tags for a list of reasons using LLM.
        
        Args:
            reasons: List of detailed error reasons
            batch_size: Number of reasons to process in each API call
            
        Returns:
            List of short error tags corresponding to each reason
        """
        if not self.openai_client:
            print("Warning: OpenAI client not available, returning generic error names")
            return [f"Error_{i+1}" for i in range(len(reasons))]
        
        error_tags = []
        
        # Process reasons in batches to avoid token limits
        for i in range(0, len(reasons), batch_size):
            batch_reasons = reasons[i:i+batch_size]
            
            prompt = f"""
            For each of the following error reasons, generate a short, descriptive error tags (4-5 words max).
            The error tag should capture the essence of the issue in a concise way. Make sure to include the 
            source of the issue as well as the type of error. 

            Format your response as a JSON list with exactly {len(batch_reasons)} items, one for each reason:
            
            Reasons:
            {json.dumps(batch_reasons, indent=2)}
            
            Example format:
            ["unrelated_hallucinated_information_returned", "tool_is_not_used", "wrong_tool_is_called", ...]

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
                        error_tags.extend(batch_names)
                    else:
                        print(f"Warning: Unexpected response format for batch {i//batch_size + 1}")
                        error_tags.extend([f"Error_{i+j+1}" for j in range(len(batch_reasons))])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON response for batch {i//batch_size + 1}")
                    error_tags.extend([f"Error_{i+j+1}" for j in range(len(batch_reasons))])
                    
            except Exception as e:
                print(f"Error generating error names for batch {i//batch_size + 1}: {e}")
                error_tags.extend([f"Error_{i+j+1}" for j in range(len(batch_reasons))])
        
        return error_tags
    
    def cluster_error_tags(self, error_tags: List[str], num_clusters: int = 10) ->  Dict[str, Any]:
        """
        Cluster error names into groups with weights using LLM.
        
        Args:
            error_tags: List of short error names
            num_clusters: Maximum number of clusters to create
            
        Returns:
            Dictionary with cluster information and weights
        """
        if not self.openai_client:
            print("Warning: OpenAI client not available, returning simple frequency clusters")
            name_counts = Counter(error_tags)
            clusters = {}
            for i, (name, count) in enumerate(name_counts.most_common(num_clusters)):
                clusters[f"Cluster_{i+1}_{name}"] = {
                    'weight': count,
                    'errors': [name] * count,
                    'description': f"Issues related to {name}"
                }
            return {'clusters': clusters, 'total_errors': len(error_tags)}
        
        # Count frequencies
        name_counts = Counter(error_tags)
        unique_names = list(name_counts.keys())
        
        prompt = f"""
        You have {len(error_tags)} error instances with {len(unique_names)} unique error types.
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
                    "error_tags": ["Error1", "Error2"],
                    "description": "Brief description of this cluster"
                }},
                "Cluster_Name_2": {{
                    "error_tags": ["Error3", "Error4"],
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
                cluster_errors = cluster_info.get('error_tags', [])
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
                'total_errors': len(error_tags),
                'total_assigned': total_assigned,
                'coverage': total_assigned / len(error_tags) if error_tags else 0
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
                'total_errors': len(error_tags),
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
            # 'score_distributions': {k: dict(v) for k, v in patterns['score_distribution'].items()},
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
        
        print("Step 3: Generating short error tags...")
        all_reasons = []
        reason_to_eval_type = {}
        
        for eval_type, reasons in patterns['reasons'].items():
            for reason in reasons:
                all_reasons.append(reason)
                reason_to_eval_type[reason] = eval_type

        error_tags = self.generate_error_tags(all_reasons)

        print("Step 4: Clustering error tags...")
        cluster_results = self.cluster_error_tags(error_tags, num_clusters)
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
                'unique_error_types': len(set(error_tags)),
                'total_reasons_analyzed': len(all_reasons)
            },
            'patterns': patterns,
            'error_tags': error_tags,
            'error_clusters': cluster_results,
            'reason_mapping': {
                'reasons_to_names': dict(zip(all_reasons, error_tags)),
                'reasons_to_eval_types': reason_to_eval_type
            },
            'llm_analysis': llm_analysis,
            'raw_imperfect_data': imperfect_evals
        }
        
        return report

    
    def visualize_cluster_results(self, cluster_results: Dict[str, Any], 
                                figsize: tuple = (12, 10),
                                save_path: Optional[str] = None) -> None:
        """
        Create simplified visualizations for cluster results showing the two main charts.
        
        Args:
            cluster_results: Dictionary containing cluster information from cluster_error_tags
            figsize: Figure size for the plots
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
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
        
        # Create vertical subplots (2 rows, 1 column)
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle('Error Cluster Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Cluster Weight Distribution (Bar Chart)
        ax1 = axes[0]
        cluster_names = list(clusters.keys())
        weights = [clusters[name]['weight'] for name in cluster_names]
        
        # Truncate long cluster names for display
        display_names = [name[:60] + '...' if len(name) > 60 else name for name in cluster_names]

        bars = ax1.barh(range(len(display_names)), weights, color=sns.color_palette("husl", len(cluster_names)))
        ax1.set_yticks(range(len(display_names)))
        ax1.set_yticklabels(display_names, fontsize=9)
        ax1.set_xlabel('Weight (Number of Errors)')
        ax1.set_title('Cluster Weights Distribution')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax1.text(weight + max(weights) * 0.01, i, f'{weight}', 
                    va='center', ha='left', fontsize=9)
        
        # 2. Cluster Coverage Pie Chart
        ax2 = axes[1]
        
        # Prepare data for pie chart
        pie_data = []
        pie_labels = []
        colors = sns.color_palette("husl", len(cluster_names) + 1)
        
        for i, (name, cluster_info) in enumerate(clusters.items()):
            pie_data.append(cluster_info['weight'])
            # Truncate labels for pie chart
            label = name[:60] + '...' if len(name) > 60 else name
            pie_labels.append(f"{label}...")

        # Add "uncovered" if coverage < 100%
        total_coverage = cluster_results.get('coverage', 1.0)
        if total_coverage < 1.0:
            uncovered = cluster_results.get('total_errors', 0) * (1 - total_coverage)
            pie_data.append(uncovered)
            pie_labels.append('Uncovered')
        
        wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title(f'Error Distribution (Coverage: {total_coverage:.1%})')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Adjust layout with padding to prevent title overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

    
    def create_interactive_drill_down(self, report: Dict[str, Any]) -> None:
        """
        Create an interactive drill-down visualization using clickable plotly charts.
        Users can click on pie chart slices to see error tags, then click on bars to see full samples.
        
        Args:
            report: Complete enhanced report from generate_enhanced_report (raw_imperfect_data contains conversations)
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import ipywidgets as widgets
            from IPython.display import display, HTML, clear_output
        except ImportError:
            print("Interactive visualization requires plotly and ipywidgets.")
            print("Install with: pip install plotly ipywidgets")
            return
        
        cluster_results = report.get('error_clusters', {})
        clusters = cluster_results.get('clusters', {})
        reason_mapping = report.get('reason_mapping', {})
        raw_data = report.get('raw_imperfect_data', [])
        
        if not clusters:
            print("No cluster data available for interactive visualization")
            return
        
        print("🎯 Interactive Error Analysis Dashboard")
        print("=" * 50)
        print("💡 Instructions:")
        print("   • Click on pie chart slices to explore clusters")
        print("   • Click on bar chart bars to see individual samples") 
        print("   • Use back buttons to navigate between levels")
        print()
        
        # Store data for use in click callbacks
        self._drill_data = {
            'clusters': clusters,
            'reason_mapping': reason_mapping,
            'raw_data': raw_data
        }
        
        # Create and show the clickable pie chart overview
        self._show_cluster_overview()
    
    def _show_cluster_overview(self) -> None:
        """Show the initial clickable pie chart overview of all clusters."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.graph_objs import FigureWidget
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("Plotly and ipywidgets not available for interactive visualization")
            return
        
        clusters = self._drill_data['clusters']
        cluster_names = list(clusters.keys())
        cluster_weights = [clusters[name]['weight'] for name in cluster_names]
        cluster_colors = px.colors.qualitative.Set3[:len(cluster_names)]
        
        # Create clickable pie chart using FigureWidget
        fig = FigureWidget()
        
        fig.add_trace(go.Pie(
            labels=[f"{name[:40]}{'...' if len(name) > 40 else ''}" for name in cluster_names],
            values=cluster_weights,
            name="Error Clusters",
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=cluster_colors),
            hovertemplate='<b>%{label}</b><br>' +
                         'Weight: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<i>Click to explore this cluster</i><extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "🎯 Error Cluster Distribution<br><sub>Click on any slice to explore that cluster</sub>",
                'x': 0.4,  # Shift title slightly left to account for legend on right
                'font': {'size': 16}
            },
            showlegend=True,
            legend=dict(
                orientation="v",  # Vertical orientation
                yanchor="middle", 
                y=0.5,
                xanchor="left",
                x=1.05  # Position legend to the right of the chart
            ),
            annotations=[dict(
                text=f"Total<br>Errors<br>{sum(cluster_weights)}",
                x=0.5, y=0.5,
                font_size=12,
                showarrow=False
            )],
            height=600,
            width=900  # Make figure wider to accommodate legend on the right
        )
        
        # Create output area for drill-down content
        self._output_area = widgets.Output()
        
        # Add click event handler
        def handle_pie_click(trace, points, selector):
            with self._output_area:
                self._output_area.clear_output()
                if points.point_inds:
                    point_index = points.point_inds[0]
                    clicked_cluster = cluster_names[point_index]
                    self._show_cluster_details_inline(clicked_cluster)
        
        fig.data[0].on_click(handle_pie_click)
        
        # Display the interactive pie chart and output area
        display(widgets.VBox([
            widgets.HTML("<h3>🎯 Interactive Error Cluster Analysis</h3>"),
            widgets.HTML("<p><i>Click on any pie slice to explore that cluster's error tags</i></p>"),
            fig,
            self._output_area
        ]))
    
    def _show_cluster_details_inline(self, cluster_name: str) -> None:
        """Show detailed view of a specific cluster with clickable error tags."""
        try:
            import plotly.graph_objects as go
            from plotly.graph_objs import FigureWidget
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("Detailed view requires plotly and ipywidgets")
            return
        
        clusters = self._drill_data['clusters']
        cluster_info = clusters[cluster_name]
        error_counts = cluster_info.get('error_counts', {})
        
        if not error_counts:
            print(f"No error tags found for cluster: {cluster_name}")
            return
        
        print(f"🎯 CLUSTER: {cluster_name}")
        print("─" * 70)
        print(f"Total Weight: {cluster_info['weight']} errors")
        print(f"Description: {cluster_info.get('description', 'No description available')}")
        print()
        
        # Create clickable horizontal bar chart for error tags
        tags = list(error_counts.keys())
        counts = list(error_counts.values())
        
        fig = FigureWidget()
        fig.add_trace(go.Bar(
            x=counts,
            y=tags,
            orientation='h',
            marker_color='lightcoral',
            text=counts,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Count: %{x}<br><i>Click to see samples</i><extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Error Tags in: {cluster_name[:60]}{'...' if len(cluster_name) > 60 else ''}<br><sub>Click on any bar to see sample conversations</sub>",
            xaxis_title="Frequency",
            yaxis_title="Error Tags",
            height=max(400, len(tags) * 40 + 150),
            margin=dict(l=200, r=50, t=100, b=50)
        )
        
        # Create output area for samples
        samples_output = widgets.Output()
        
        # Add click event handler for bars
        def handle_bar_click(trace, points, selector):
            with samples_output:
                samples_output.clear_output()
                if points.point_inds:
                    point_index = points.point_inds[0]
                    clicked_tag = tags[point_index]
                    self._show_tag_samples_inline(cluster_name, clicked_tag)
        
        fig.data[0].on_click(handle_bar_click)
        
        # Create back button
        back_button = widgets.Button(
            description="⬅️ Back to Cluster Overview",
            layout=widgets.Layout(width='250px'),
            style={'description_width': 'initial'},
            button_style='info'
        )
        
        def go_back(b):
            with self._output_area:
                self._output_area.clear_output()
                print("👆 Click on the pie chart above to explore clusters")
        
        back_button.on_click(go_back)
        
        # Display everything
        display(widgets.VBox([
            back_button,
            widgets.HTML("<p><i>Click on any bar to see full conversation samples for that error tag</i></p>"),
            fig,
            samples_output
        ]))

    def _show_tag_samples_inline(self, cluster_name: str, tag: str) -> None:
        """Show full conversation samples for a specific error tag."""
        reason_mapping = self._drill_data['reason_mapping']
        raw_data = self._drill_data['raw_data']
        
        print(f"🔍 SAMPLES FOR ERROR TAG: {tag}")
        print("═" * 80)
        print(f"📁 Cluster: {cluster_name}")
        print(f"🏷️  Tag: {tag}")
        print()
        
        # Find evaluations related to this tag
        reasons_to_names = reason_mapping.get('reasons_to_names', {})
        
        # Find reasons that map to this tag
        related_reasons = [reason for reason, mapped_tag in reasons_to_names.items() if mapped_tag == tag]
        
        if not related_reasons:
            print("❌ No related reasons found for this tag")
            return
        
        # Find evaluations that have these reasons (conversations are already in raw_data)
        matching_evaluations = []
        for eval_data in raw_data:
            results = eval_data.get('results', {})
            for eval_name, eval_result in results.items():
                if eval_name.lower() in eval_result:
                    eval_field_name = eval_name.lower()
                else:
                    eval_field_name = self._camel_to_snake(eval_name)
                reason_key = f'{eval_field_name}_reason'
                reason = eval_result.get(reason_key, '')
                if reason in related_reasons:
                    matching_evaluations.append(eval_data)
                    break  # Found a match for this evaluation, no need to check other eval types
        
        if not matching_evaluations:
            print("❌ No matching samples found")
            return
        
        print(f"📊 Found {len(matching_evaluations)} samples with this error tag")
        print(f"📋 Showing first {min(3, len(matching_evaluations))} samples:\n")
        
        # Show up to 3 samples
        for i, eval_data in enumerate(matching_evaluations[:3]):
            print(f"📝 SAMPLE {i+1} of {len(matching_evaluations)}")
            print("─" * 60)
            
            thread_id = eval_data.get('thread_id', 'Unknown')
            conversation = eval_data.get('conversation', {})
            
            # Show well-formatted query-response conversation
            self._display_formatted_conversation_from_data(conversation, thread_id)
            
            # Show evaluation details for this specific tag
            results = eval_data.get('results', {})
            cluster_eval_info = {}
            
            for eval_name, eval_result in results.items():
                if eval_name.lower() in eval_result:
                    eval_field_name = eval_name.lower()
                else:
                    eval_field_name = self._camel_to_snake(eval_name)
                reason_key = f'{eval_field_name}_reason'
                reason = eval_result.get(reason_key, '')
                if reason in related_reasons:
                    score_key = f'{eval_field_name}_score'
                    score = eval_result.get(score_key, eval_result.get(eval_field_name, 'N/A'))
                    result_key = f'{eval_field_name}_result'
                    result = eval_result.get(result_key, 'N/A')
                    cluster_eval_info[eval_name] = {
                        'score': score,
                        'result': result,
                        'reason': reason,
                        'tag': tag
                    }
            
            # Show cluster-based evaluation details
            if cluster_eval_info:
                print("🎯 EVALUATION ANALYSIS")
                print("─" * 50)
                print(f"🏷️  Error Tag: {tag}")
                print(f"🎯 Related Cluster: {cluster_name}")
                print()
                
                for eval_name, eval_info in cluster_eval_info.items():
                    print(f"📊 {eval_name} Evaluation:")
                    
                    # Format score nicely
                    score = eval_info['score']
                    if isinstance(score, (int, float)):
                        score_display = f"{score:.2f}" if isinstance(score, float) else str(score)
                        # Add visual indicator for score
                        if isinstance(score, (int, float)) and score <= 2:
                            score_display += " ❌ (Poor)"
                        elif isinstance(score, (int, float)) and score <= 3:
                            score_display += " ⚠️  (Fair)" 
                        elif isinstance(score, (int, float)) and score >= 4:
                            score_display += " ✅ (Good)"
                    else:
                        score_display = str(score)
                    
                    print(f"   • Score: {score_display}")
                    print(f"   • Result: {eval_info['result']}")
                    
                    # Format reason nicely (wrap long reasons)
                    reason = eval_info['reason']
                    if len(reason) > 80:
                        # Wrap long reasons
                        reason_words = reason.split()
                        reason_lines = []
                        current_line = []
                        current_length = 0
                        
                        for word in reason_words:
                            if current_length + len(word) > 75:
                                if current_line:
                                    reason_lines.append(" ".join(current_line))
                                    current_line = [word]
                                    current_length = len(word)
                            else:
                                current_line.append(word)
                                current_length += len(word) + 1
                        
                        if current_line:
                            reason_lines.append(" ".join(current_line))
                        
                        print(f"   • Reason:")
                        for line in reason_lines:
                            print(f"     {line}")
                    else:
                        print(f"   • Reason: {reason}")
                    print()
            else:
                print("⚠️  No evaluation details found for this sample")
                print()
            
            print("═" * 70)
            print()
                
        if len(matching_evaluations) > 3:
            remaining = len(matching_evaluations) - 3
            print(f"... and {remaining} more samples with this error tag.")
        
        # Add back button
        try:
            import ipywidgets as widgets
            from IPython.display import display
            
            back_button = widgets.Button(
                description="⬅️ Back to Error Tags",
                layout=widgets.Layout(width='200px'),
                style={'description_width': 'initial'},
                button_style='warning'
            )
            
            def go_back_to_tags(b):
                with self._output_area:
                    self._output_area.clear_output()
                    self._show_cluster_details_inline(cluster_name)
            
            back_button.on_click(go_back_to_tags)
            display(back_button)
            
        except ImportError:
            print("📌 Use the interactive interface above to navigate back")

    def _display_formatted_conversation_from_data(self, conversation: Dict[str, Any], thread_id: str) -> None:
        """
        Display a well-formatted conversation from conversation data in raw_imperfect_data.
        
        Args:
            conversation: Conversation data from raw_imperfect_data
            thread_id: Thread ID for reference
        """
        print("💬 CONVERSATION DETAILS")
        print("─" * 50)
        print(f"🆔 Thread ID: {thread_id}")
        print()
        
        # Extract query and response from conversation data
        query = conversation.get('query', 'N/A')
        response = conversation.get('response', 'N/A')
        
        # Display query
        print("🔍 USER QUERY:")
        if query and query != 'N/A':
            if isinstance(query, list):
                query_text = self._extract_texts_from_message(query)
            else:
                query_text = str(query)
            print(f"   {query_text}")
        else:
            print("   [Not available in conversation data]")
        print()
        
        # Display response
        print("🤖 SYSTEM RESPONSE:")
        if response and response != 'N/A':
            if isinstance(response, list):
                response_text = self._extract_texts_from_message(response)
            else:
                response_text = str(response)
            
            # Format long responses nicely
            if len(response_text) > 200:
                # Break long responses into readable chunks
                words = response_text.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) > 80:  # 80 chars per line
                        if current_line:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            lines.append(word)  # Word too long, add as is
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1  # +1 for space

                # Append any remaining words as the last line
                if current_line:
                    lines.append(" ".join(current_line))
                
                for line in lines:
                    print(f"   {line}")
            else:
                print(f"   {response_text}")
        else:
            print("   [Not available in conversation data]")
        print()

    def _display_formatted_conversation(self, eval_sample: Dict[str, Any]) -> None:
        """
        Display a well-formatted conversation showing query-response pairs.
        
        Args:
            eval_sample: Sample data containing conversation information
        """
        print("💬 CONVERSATION DETAILS")
        print("─" * 50)
        
        # Extract basic information
        thread_id = eval_sample.get('thread_id', 'Unknown')
        print(f"🆔 Thread ID: {thread_id}")
        

        # if we load evaluation from foundry evaluation jsonl, the format is flat.
        # critic_agent.evaluate() returns a nested structure
        if "conversation" in eval_sample:
            conversation = eval_sample["conversation"]
        else:
            conversation = eval_sample

        query = None
        if isinstance(eval_sample, dict):
            # Common patterns for queries in evaluation data
            query = (conversation.get('inputs.query') or 
                    conversation.get('query') or 
                    conversation.get('user_query') or 
                    conversation.get('input') or
                    conversation.get('prompt') or
                    str(conversation) if conversation else None)
        
        # Try to extract response from outputs
        response = None
        if isinstance(conversation, dict):
            # Common patterns for responses in evaluation data
            response = (conversation.get('inputs.response') or 
                       conversation.get('response') or 
                       conversation.get('output') or 
                       conversation.get('result') or
                       conversation.get('completion') or
                       str(conversation) if conversation else None)

        # Display query
        if query:
            print("🔍 USER QUERY:")
            print(f"   {self._extract_texts_from_message(query)}")
            print()
        else:
            print("🔍 USER QUERY: [Not available in sample data]")
            print()
        
        # Display response
        if response:
            print("🤖 RESPONSE:")
            # Format long responses nicely
            response = self._extract_texts_from_message(response)
            if isinstance(response, str) and len(response) > 200:
                # Break long responses into readable chunks
                words = response.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) > 80:  # 80 chars per line
                        if current_line:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            lines.append(word)  # Word too long, add as is
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1  # +1 for space

                # Append any remaining words as the last line
                if current_line:
                    lines.append(" ".join(current_line))

            print(response)
            print()
        else:
            print("🤖 SYSTEM RESPONSE: [Not available in sample data]")
            print()

    def _extract_texts_from_message(self, message: Any) -> str:
        """
        Extracts text content from various message formats.
        
        Args:
            message: Can be a list of dicts, string, or other format
            
        Returns:
            Extracted text as string
        """
        if isinstance(message, str):
            return message
        elif isinstance(message, list):
            texts = []
            for entry in message:
                if isinstance(entry, dict):
                    if "role" in entry:
                        role = entry["role"]
                        texts.append(f"\n{role}:\n")
                    if 'content' in entry:
                        content = entry['content']
                        if isinstance(content, list):
                            for content_item in content:
                                if isinstance(content_item, dict) and content_item.get('type') == 'text':
                                    texts.append(content_item.get('text', ''))
                                elif isinstance(content_item, dict) and content_item.get('type') == 'tool_result':
                                    texts.append(str(content_item.get('tool_result', '')))
                        elif isinstance(content, str):
                            texts.append(content)
                    elif 'text' in entry:
                        texts.append(entry['text'])
                elif isinstance(entry, str):
                    texts.append(entry)
            return "\n".join(texts)
        else:
            return str(message)