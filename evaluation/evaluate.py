#!/usr/bin/env python

import sys
import os
sys.path.append(os.getcwd())
import argparse
import asyncio
import json
import time
import datetime
from typing import Dict, Any, List, Optional

from src.evaluator import Evaluator


async def main():

    parser = argparse.ArgumentParser(description="Evaluation Tool")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file')
    parser.add_argument('--task', type=str, required=True, choices=['math', 'qa'], help='Type of evaluation task')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM for equivalence evaluation')
    parser.add_argument('--api_base_url', type=str, default=None, help='Base URL of the LLM API')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the LLM model used for evaluation')
    parser.add_argument('--concurrent_limit', type=int, default=50, help='Maximum number of concurrent evaluations')
    parser.add_argument('--timeout', type=int, default=1800, help='Total evaluation timeout in seconds')
    parser.add_argument('--counts', type=int, default=100, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    try:
        print(f"Model output file path: {args.output_path}")
        
        # Check if file exists
        if not os.path.exists(args.output_path):
            raise FileNotFoundError(f"Output file does not exist: {args.output_path}")
        
        # Load data
        print("Loading data...")
        start_time = time.time()
        try:
            # 这行代码的意思是：先用os.path.splitext分离文件名和扩展名，取扩展名file_ext，然后将其转换为小写，方便后续判断文件类型。
            _, file_ext = os.path.splitext(args.output_path)
            file_ext = file_ext.lower()
            
            with open(args.output_path, 'r', encoding='utf-8') as f:
                if file_ext == '.json':
                    # Standard JSON format
                    data = json.load(f)
                elif file_ext in ['.jsonl', '.txt']:
                    # JSON Lines or TXT format (line-by-line JSON)
                    data = []
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            data.append(json.loads(line))
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
            data = data[:args.counts]
            print(f"Data loading completed. Total {len(data)} samples. Time taken: {time.time() - start_time:.2f} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {str(e)}")
        
        # Create evaluator
        evaluator = Evaluator(
            task_type=args.task,
            output_path=args.output_path,  
            use_llm=args.use_llm,
            api_base_url=args.api_base_url,
            model_name=args.model_name,
            concurrent_limit=args.concurrent_limit
        )
        
        # Show output path information
        print(f"Detailed metrics path: {evaluator.output_metrics_path}")
        print(f"Overall metrics path: {evaluator.output_metrics_overall_path}")
        
        # # Set timeout
        # try:
        #     overall_metrics = await asyncio.wait_for(evaluator.run(data), timeout=args.timeout)
        # except asyncio.TimeoutError:
        #     print(f"Warning: Evaluation timed out ({args.timeout} seconds)")
        #     overall_metrics = {"status": "timeout"}
        overall_metrics = await evaluator.run(data, timeout=args.timeout)
        
        print(f"Evaluation completed. Total time: {time.time() - start_time:.2f} seconds")
        
        return overall_metrics
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run main function
    results = asyncio.run(main())
    
    # Print result summary
    if results.get("status") in ["error", "timeout"]:
        print(f"\n===== Evaluation Not Completed: {results.get('status')} =====")
        if results.get("message"):
            print(f"Reason: {results.get('message')}")
        sys.exit(1)
    
    print("\n===== Evaluation Summary =====")
    for key, value in results.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"{key}: {value:.4f}")
        elif key not in ['domain_metrics']: 
            print(f"{key}: {value}")
    
    # Exit successfully
    sys.exit(0)
