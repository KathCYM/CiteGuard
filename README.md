<h2>CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation</h2>

<p>
<a href="https://www.arxiv.org/abs/2510.17853"><strong>Paper Link</strong></a>
Large Language Models (LLMs) have emerged as promising assistants for scientific writing. However, there have been concerns regarding the quality and reliability of the generated text, one of which is the citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which is assessing whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves the prior baseline by 12.3%, and achieves up to 65.4% accuracy on the CiteME benchmark, on par with human-level performance (69.7%). It also enables the identification of alternative but valid citations.
</p>

## CiteGuardAgent

### Environment variables

CiteAgent requires following environment variables to function properly:
- `S2_API_KEY`: Your semantic scholar api key

Any of the following depending on which model/platform you intend to use.
- `OPENAI_API_KEY`: Your openai api key (for openai models)
- `ANTHROPIC_API_KEY`: Your anthropic api key (for claude models)
- `TOGETHER_API_KEY`: Your together api key (for llama models)
- `DEEPSEEK_API_KEY`: Your deepseek api key (for deepseek models)

### Run CiteME evaluation
1. Install the required python packages listed in the `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```

2. Download the dataset from [citeme.ai](https://www.citeme.ai) and place it in the project folder as `DATASET.csv`. You will need to convert from .tsv to .csv if needed.

3. Run the following command. 
   ```
   python src/main.py --dataset <path to citeme.csv> --model_name <model name> 
   ```

### Run your own citation search
1. Follow the same setup instruction, and run the following command:
   ```
   python src/main.py  --model_name <model name> --excerpt <str containing the excerpt, replace where you want the citation with [CITATION]> --skip_citations <str of list of citations to skip, seperated by commas>
   ```

### Run with your local model (Ollama)
1. To run with your local model using ollama, simply add the `--local_model` flag and use the ollama model name you used to create the model.

### Flask Server
1. To start a flask server, simply run the command
   ```
   python app.py
   ```
2. Use either the UI at http://127.0.0.1:5000/ or the following command
   ```
   curl -X POST http://127.0.0.1:5000/run_stream      -H "Content-Type: application/json"      -d '{
           "model_name": <model name>,
           "excerpt": <str containing the excerpt, replace where you want the citation to [CITATION]>
         }' 
   ```
## 🪪 License <a name="license"></a>
Code: MIT. Check `LICENSE`.
Dataset: CC-BY-4.0. Check `LICENSE_DATASET`.

If you find our code useful, please cite our paper:
```
@misc{choi2025citeguardfaithfulcitationattribution,
      title={CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation}, 
      author={Yee Man Choi and Xuehang Guo and Yi R. Fung and Qingyun Wang},
      year={2025},
      eprint={2510.17853},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2510.17853}, 
}
```
