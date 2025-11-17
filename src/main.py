import os
import json
import pandas as pd
from rich.console import Console
from retriever.agent import (
    LLMSelfAskAgentPydantic,
    LLMNoSearch,
    OutputNoRead,
    OutputSearchOnly,
    Output,
)
from utils.str_matcher import find_match_psr, find_multi_match_psr
from utils.tokens import num_tokens_from_string
from datetime import datetime
from time import time
from rich.progress import track
from retriever.llm_base import DEFAULT_TEMPERATURE
import argparse
from run_main import run

def main():
    parser = argparse.ArgumentParser(description="Run citation retrieval agent.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset CSV file. If omitted, runs in single input mode.")
    parser.add_argument("--result_path", type=str, default="results.json",
                        help="Path to result JSON file (default: results.json).")
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument("--local_model", action="store_true", help="Use local model if set.")

    # single data point inputs
    parser.add_argument("--id", type=str, default="manual",
                        help="Unique ID for single input mode (default: manual).")
    parser.add_argument("--source_paper_title", type=str, default=None,
                        help="Optional source paper title.")
    parser.add_argument("--target_paper_title", type=str, default=None, help="Target paper title if running evaluation, separated by [TITLE_SEPARATOR] if needed.")
    parser.add_argument("--excerpt", type=str, help="Citation excerpt text.")
    parser.add_argument("--year", type=int, default=2025, help="Publication year of source paper which contains the excerpt (default: 2025).")
    parser.add_argument("--skip_citations", type=str, default="",
                        help="Comma-separated list of found citations for the excerpt (used in single input mode).")

    # parameters
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Temperature to use for the LLM (default: 0.95).")
    
    args = parser.parse_args()
    console = Console()
    results = run(args, console)
    return results

if __name__ == "__main__":
    main()
