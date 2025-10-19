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

# -- Modify the following variables as needed --
TITLE_SEPERATOR = "[TITLE_SEPARATOR]"
RESULT_FILE_NAME = f"few-shot-tool-<model_name>.json"
INCREMENTAL_SAVE = True

metadata = {
    "model": "<model_name>",
    "temperature": DEFAULT_TEMPERATURE,
    "executor": "LLMSelfAskAgentPydantic",
    "search_provider": "SemanticScholarSearchProvider",
    "prompt_name": "few_shot_tool",  # See prompt names in retriever/prompt_templates
    "actions": "search_relevance,search_citation_count,select,find_in_text,ask_for_more_context,search_text_snippet",
    "search_limit": 10,
    "threshold": 0.8,
    "execution_date": datetime.now().isoformat(),
    "only_open_access": False,
    "dataset_split": "all",
    "use_web_search": False,
    "max_actions": 15,
}
# -- NO USER EDITABLE CODE BELOW THIS LINE --

console = Console()

## Load the dataset
c = pd.read_csv("./DATASET.csv", sep=",")
c.set_index("id", inplace=True)
if metadata["dataset_split"] == "all":
    pass
elif metadata["dataset_split"] == "test":
    c = c[c["split"] == 'test']
elif metadata["dataset_split"] == "train":
    c = c[c["DevSet"] == 'train']
else:
    raise ValueError("Invalid dataset split")

## Select executor
if metadata["executor"] == "LLMSelfAskAgentPydantic":
    # Select action space
    if metadata["actions"] == "search_relevance,search_citation_count,read,select,find_in_text,ask_for_more_context,search_text_snippet":
        pdo = Output
        print("using Ouput pdo")
    if metadata["actions"] == "search_relevance,search_citation_count,select,find_in_text,ask_for_more_context,search_text_snippet":
        pdo = OutputNoRead
        print("using OuputNoRead pdo")
    elif metadata["actions"] == "search_relevance,search_citation_count,select":
        pdo = OutputSearchOnly
        print("using OutputSearchOnly pdo")
    else:
        raise ValueError("Invalid actions")
    agent = LLMSelfAskAgentPydantic(
        only_open_access=metadata["only_open_access"],
        search_limit=metadata["search_limit"],
        model_name=metadata["model"],
        temperature=metadata["temperature"],
        use_web_search=metadata["use_web_search"],
        prompt_name=metadata["prompt_name"],
        pydantic_object=pdo,
    )
elif metadata["executor"] == "LLMNoSearch":
    metadata["actions"] = None
    metadata["search_provider"] = None
    metadata["search_limit"] = 0
    metadata["use_web_search"] = False
    metadata["max_actions"] = 0
    assert metadata["prompt_name"].endswith("no_search"), "Only no search prompt are supported when executor is LLMNoSearch"
    agent = LLMNoSearch(
        model_name=metadata["model"],
        temperature=metadata["temperature"],
        prompt_name=metadata["prompt_name"],
    )
else:
    raise ValueError("Invalid executor")

console.log(f"AGENT BACKBONE: [bold green]{metadata['model']}")
console.log(f"PROMPT:         [bold green]{metadata['prompt_name']}")
console.log(f"ACTIONS:        [bold blue]{metadata['actions']}")

results = []
if os.path.exists(RESULT_FILE_NAME):
    with open(RESULT_FILE_NAME, "r") as f:
        existing_data = json.load(f)
        old_results = existing_data.get("results", [])
        results += old_results

processed_ids = [item["id"] for item in results]
print(f"Already processed: len{processed_ids}")

for cid, citation in track(
    c.iterrows(), description="Processing citations", total=len(c), console=console
):
    if cid in processed_ids:
        continue
    agent.reset([citation["source_paper_title"]])
    
    start_time = time()
    citation_text = citation["excerpt"]
    src_paper_title = citation["source_paper_title"]
    target_titles = citation["target_paper_title"].split(TITLE_SEPERATOR)
    year = int(citation["year"] if not pd.isna(citation["year"]) else 2024)
    result_data = {
        "id": cid,
        "cited_paper_titles": target_titles,
        "excerpt": citation_text,
    }
    try:
        selection = agent(citation_text, f"{year}", src_paper_title=src_paper_title, max_actions=metadata["max_actions"])
        result_data["selected"] = selection.model_dump()
        is_in_search = find_multi_match_psr(
            sum(agent.paper_buffer, []), target_titles, threshold=metadata["threshold"]
        )
        result_data["is_in_search"] = is_in_search[1] if is_in_search else None
        correct = (
            find_match_psr(target_titles, selection.title, metadata["threshold"])
            is not None
        )
        result_data["is_correct"] = correct
        result_data["status"] = "success"
        result_data["error"] = None

        console.log(
            f"{cid:3d}: {target_titles[0]}\n"
            + f"     [{'green' if correct else 'red'}]{selection.title}\n"
            + f"     [black]search: {'✅' if is_in_search is not None else '❌'}, selection: {'✅' if correct else '❌'}"
        )
    except Exception as e:
        result_data["selected"] = None
        result_data["is_in_search"] = None
        result_data["is_correct"] = False
        result_data["status"] = "error"
        result_data["error"] = str(e)
        console.log(f"[red]{cid:3d}: {target_titles[0]}\n     [bold]{e}")

    result_data["duration"] = time() - start_time
    result_data["papers"] = agent.get_paper_buffer()
    result_data["history"] = agent.get_history(ignore_system_messages=False)
    if metadata["model"].startswith("gpt"):
        result_data["tokens"] = {
            "input": sum(
                [
                    num_tokens_from_string(m["content"], metadata["model"])
                    for m in result_data["history"]
                    if m["role"] != "assistant"
                ]
            ),
            "output": sum(
                [
                    num_tokens_from_string(m["content"], metadata["model"])
                    for m in result_data["history"]
                    if m["role"] == "assistant"
                ]
            ),
        }

    results.append(result_data)

    if INCREMENTAL_SAVE:
        with open(RESULT_FILE_NAME, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=4)

with open(RESULT_FILE_NAME, "w") as f:
    json.dump({"metadata": metadata, "results": results}, f, indent=4)
