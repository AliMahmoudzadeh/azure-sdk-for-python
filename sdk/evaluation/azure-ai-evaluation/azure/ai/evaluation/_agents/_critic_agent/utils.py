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


def load_evaluation_data_with_samples(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from JSONL file  (output from foundry evaluation) and return 
    both processed evaluations and original samples.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        - evaluations: List of evaluation dictionaries in the expected format
    """
    evaluations = []
    conversations = []
    
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
                        conversation = {}
                        conversation['thread_id'] = thread_id
                        conversation['query'] = data['inputs.query']
                        conversation['response'] = data['inputs.response']
                        conversation['tool_definitions'] = data['inputs.tool_definitions']
                        conversations.append(conversation)
                        
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
                                'results': results,
                                'conversation': conversation
                            })
                    
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return []

    print(f"Loaded {len(evaluations)} evaluations from JSONL file")
    return evaluations