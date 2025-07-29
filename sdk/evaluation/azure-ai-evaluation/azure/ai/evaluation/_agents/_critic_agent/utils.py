import pandas as pd
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import openai
import os
import re

def camel_to_snake( name: str) -> str:
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

def load_evaluation_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of evaluation dictionaries in the expected format
    """
    evaluations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        
                        # Extract thread ID (use line_number if available, otherwise create one)
                        thread_id = data.get('line_number', line_num)
                        if thread_id is None:
                            thread_id = line_num
                        thread_id = f"line_{thread_id}"
                        
                        # Parse outputs.xyz format
                        results = {}
                        
                        # Group outputs by evaluation type
                        output_groups = defaultdict(dict)
                        
                        for key, value in data.items():
                            if key.startswith('outputs.'):
                                # Parse key like "outputs.intent_resolution.intent_resolution"
                                parts = key.split('.')
                                if len(parts) >= 3:
                                    eval_type = parts[1]  # e.g., "intent_resolution"
                                    metric_name = '.'.join(parts[2:])  # e.g., "intent_resolution" or "intent_resolution_result"
                                    
                                    # Convert eval_type to CamelCase for consistency
                                    eval_type_camel = ''.join(word.capitalize() for word in eval_type.split('_'))
                                    
                                    output_groups[eval_type_camel][metric_name] = value
                        
                        # Convert to expected format
                        for eval_type, metrics in output_groups.items():
                            # Find the base metric name (the one that matches the eval type)
                            base_name = camel_to_snake(eval_type)
                            score_key = None
                            result_key = None
                            reason_key = None
                            threshold_key = None
                            
                            # Look for score, result, reason, and threshold keys
                            for metric_key in metrics.keys():
                                if metric_key == base_name :
                                    score_key = metric_key
                                elif metric_key.endswith('_result'):
                                    result_key = metric_key
                                elif metric_key.endswith('_reason'):
                                    reason_key = metric_key
                                elif metric_key.endswith('_threshold'):
                                    threshold_key = metric_key
                            
                            # Build the result structure
                            result_data = {}
                            
                            if score_key and score_key in metrics:
                                result_data[score_key] = metrics[score_key]
                            if result_key and result_key in metrics:
                                result_data[result_key] = metrics[result_key]
                            if reason_key and reason_key in metrics:
                                result_data[reason_key] = metrics[reason_key]
                            if threshold_key and threshold_key in metrics:
                                result_data[threshold_key] = metrics[threshold_key]

                            # Add any other metrics
                            for metric_key, metric_value in metrics.items():
                                if metric_key not in [score_key, result_key, reason_key, threshold_key]:
                                    result_data[metric_key] = metric_value
                            
                            if result_data:
                                results[eval_type] = result_data
                        
                        if results:
                            evaluations.append({
                                'thread_id': thread_id,
                                'results': results
                            })
                    
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return []
    
    print(f"Loaded {len(evaluations)} evaluations from JSONL file")
    return evaluations

def load_evaluation_data_parquet(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from parquet file (legacy function).
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        List of evaluation dictionaries
    """
    try:
        df = pd.read_parquet(file_path)
        # Assuming the evaluations are in a column named 'evaluations' or similar
        # Adjust this based on your actual data structure
        if 'evaluations' in df.columns:
            return df['evaluations'].tolist()
        else:
            # If the data structure is different, you might need to adjust this
            print(f"Available columns: {df.columns.tolist()}")
            return df.to_dict('records')
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def display_report(report: Dict[str, Any]) -> None:
    """
    Display the error analysis report in a formatted way.
    
    Args:
        report: Report dictionary from ErrorAnalyzer.generate_report()
    """
    print("=" * 80)
    print("ERROR ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Imperfect Evaluations: {summary['imperfect_evaluations']}")
    print(f"  Total Issues Found: {summary['total_issues']}")
    print(f"  Filter Mode: {summary['filter_mode']}")
    if summary['total_evaluations'] > 0:
        print(f"  Issue Rate: {summary['imperfect_evaluations']/summary['total_evaluations']:.2%}")
    
    # Patterns
    patterns = report['patterns']
    print(f"\nEVALUATION TYPE FREQUENCY:")
    for eval_type, count in sorted(patterns['evaluation_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {eval_type}: {count} issues")
    
    print(f"\nSCORE DISTRIBUTIONS:")
    for eval_type, scores in patterns['score_distribution'].items():
        print(f"  {eval_type}:")
        for score, count in sorted(scores.items()):
            print(f"    Score {score}: {count} occurrences")
    
    # LLM Analysis
    if report['llm_analysis']:
        print(f"\nLLM ANALYSIS:")
        print("-" * 40)
        print(report['llm_analysis'])
    
    print("\n" + "=" * 80)

def display_enhanced_report(report: Dict[str, Any]) -> None:
    """
    Display the enhanced error analysis report with clustering information.
    
    Args:
        report: Report dictionary from ErrorAnalyzer.generate_enhanced_report()
    """
    print("=" * 80)
    print("ENHANCED ERROR ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Imperfect Evaluations: {summary['imperfect_evaluations']}")
    print(f"  Total Issues Found: {summary['total_issues']}")
    print(f"  Filter Mode: {summary['filter_mode']}")
    print(f"  Unique Error Types: {summary.get('unique_error_types', 'N/A')}")
    print(f"  Total Reasons Analyzed: {summary.get('total_reasons_analyzed', 'N/A')}")
    if summary['total_evaluations'] > 0:
        print(f"  Issue Rate: {summary['imperfect_evaluations']/summary['total_evaluations']:.2%}")
    
    # Error Clusters
    if 'error_clusters' in report:
        clusters = report['error_clusters']
        print(f"\nERROR CLUSTERS:")
        print(f"  Total Error Instances: {clusters.get('total_errors', 0)}")
        print(f"  Coverage: {clusters.get('coverage', 0):.1%}")
        print(f"\n  Top Clusters by Weight:")
        
        # Sort clusters by weight
        sorted_clusters = sorted(
            clusters.get('clusters', {}).items(), 
            key=lambda x: x[1]['weight'], 
            reverse=True
        )
        
        for i, (cluster_name, cluster_info) in enumerate(sorted_clusters[:10], 1):
            weight = cluster_info['weight']
            errors = cluster_info.get('errors', [])
            description = cluster_info.get('description', 'No description')
            
            print(f"\n  {i}. {cluster_name}")
            print(f"     Weight: {weight} ({weight/clusters.get('total_errors', 1):.1%})")
            print(f"     Description: {description}")
            
            # Show error counts if available
            if 'error_counts' in cluster_info:
                error_counts = cluster_info['error_counts']
                print(f"     Error Types:")
                for error_name, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"       - {error_name}: {count}")
            else:
                print(f"     Unique Errors: {len(set(errors))}")
    
    # Traditional patterns (for comparison)
    patterns = report['patterns']
    print(f"\nEVALUATION TYPE FREQUENCY:")
    for eval_type, count in sorted(patterns['evaluation_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {eval_type}: {count} issues")
    
    print(f"\nSCORE DISTRIBUTIONS:")
    for eval_type, scores in patterns['score_distribution'].items():
        print(f"  {eval_type}:")
        for score, count in sorted(scores.items()):
            print(f"    Score {score}: {count} occurrences")
    
    # LLM Analysis
    if report['llm_analysis']:
        print(f"\nLLM ANALYSIS:")
        print("-" * 40)
        print(report['llm_analysis'])
    
    print("\n" + "=" * 80)

def load_evaluation_data_with_samples(file_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load evaluation data from JSONL file and return both processed evaluations and original samples.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Tuple of (evaluations, original_samples):
        - evaluations: List of evaluation dictionaries in the expected format
        - original_samples: List of original sample data with full conversation content
    """
    evaluations = []
    original_samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        
                        # Extract thread ID (use line_number if available, otherwise create one)
                        thread_id = data.get('line_number', line_num)
                        if thread_id is None:
                            thread_id = line_num
                        thread_id = f"line_{thread_id}"
                        
                        # Store original sample with thread_id
                        original_sample = data.copy()
                        original_sample['thread_id'] = thread_id
                        original_samples.append(original_sample)
                        
                        # Parse outputs.xyz format for evaluations
                        results = {}
                        
                        # Group outputs by evaluation type
                        output_groups = defaultdict(dict)
                        
                        for key, value in data.items():
                            if key.startswith('outputs.'):
                                # Parse key like "outputs.intent_resolution.intent_resolution"
                                parts = key.split('.')
                                if len(parts) >= 3:
                                    eval_type = parts[1]  # e.g., "intent_resolution"
                                    metric_name = '.'.join(parts[2:])  # e.g., "intent_resolution" or "intent_resolution_result"
                                    
                                    # Convert eval_type to CamelCase for consistency
                                    eval_type_camel = ''.join(word.capitalize() for word in eval_type.split('_'))
                                    
                                    output_groups[eval_type_camel][metric_name] = value
                        
                        # Convert to expected format
                        for eval_type, metrics in output_groups.items():
                            # Find the base metric name (the one that matches the eval type)
                            base_name = eval_type.lower()
                            score_key = None
                            result_key = None
                            reason_key = None
                            threshold_key = None
                            
                            # Look for score, result, reason, and threshold keys
                            for metric_key in metrics.keys():
                                metric_lower = metric_key.lower()
                                if metric_lower == base_name or metric_lower in eval_type.lower():
                                    score_key = metric_key
                                elif metric_key.endswith('_result'):
                                    result_key = metric_key
                                elif metric_key.endswith('_reason'):
                                    reason_key = metric_key
                                elif metric_key.endswith('_threshold'):
                                    threshold_key = metric_key
                            
                            # Build the result structure
                            result_data = {}
                            
                            if score_key and score_key in metrics:
                                result_data[score_key] = metrics[score_key]
                            if result_key and result_key in metrics:
                                result_data[result_key] = metrics[result_key]
                            if reason_key and reason_key in metrics:
                                result_data[reason_key] = metrics[reason_key]
                            if threshold_key and threshold_key in metrics:
                                result_data[threshold_key] = metrics[threshold_key]
                            
                            # Add any other metrics
                            for metric_key, metric_value in metrics.items():
                                if metric_key not in [score_key, result_key, reason_key, threshold_key]:
                                    result_data[metric_key] = metric_value
                            
                            if result_data:
                                results[eval_type] = result_data
                        
                        if results:
                            evaluations.append({
                                'thread_id': thread_id,
                                'results': results
                            })
                    
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return [], []
    
    print(f"Loaded {len(evaluations)} evaluations and {len(original_samples)} original samples from JSONL file")
    return evaluations, original_samples