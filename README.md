<h2>CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation</h2>

<p>
<a href="#"><strong>Paper Link (Coming Soon)</strong></a>
Large Language Models (LLMs) have emerged as promising assistants for scientific writing. However, there have been concerns regarding the quality and reliability of the generated text, one of which is the citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which is assessing whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves the prior baseline by 12.3%, and achieves up to 65.4% accuracy on the CiteME benchmark, on par with human-level performance (69.7%). It also enables the identification of alternative but valid citations.
</p>

## CiteGuardAgent

### Environment variables

CiteAgent requires following environment variables to function properly:
- `S2_API_KEY`: Your semantic scholar api key

Any of the following depending on which model/platform you intend to use.
- `OPENAI_API_KEY`: Your openai api key (for gpt-4 models)
- `ANTHROPIC_API_KEY`: Your anthropic api key (for claude models)
- `TOGETHER_API_KEY`: Your together api key (for llama models)

### Run
1. Install the required python packages listed in the `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```

2. Download the dataset from [citeme.ai](https://www.citeme.ai) and place it in the project folder as `DATASET.csv`. You will need to convert from .tsv to .csv if needed.

3. Update the model\_name in the `main.py` file and run the following command. 
   ```
   python src/main.py
   ```

## 🪪 License <a name="license"></a>
Code: MIT. Check `LICENSE`.
Dataset: CC-BY-4.0. Check `LICENSE_DATASET`.
