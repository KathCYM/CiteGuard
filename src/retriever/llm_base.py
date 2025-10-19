import os
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_TEMPERATURE = 0.95

def get_model_by_name(model_name: str, temperature: float = DEFAULT_TEMPERATURE):
    if "gpt-oss" in model_name.lower():
        return ChatTogether(
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            temperature=temperature,
            model=model_name,
        )
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
        # return AzureOpenAI(model=model_name, temperature=temperature)
    if model_name.startswith("claude-"):
        return ChatAnthropic(
            temperature=temperature,
            model_name=model_name,
            timeout=60*10,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    if "llama" in model_name.lower() or "phi" in model_name.lower() or "mistral" in model_name.lower():
        return ChatTogether(
            # together_api_key="YOUR_API_KEY",
            temperature=temperature,
            model=model_name,
        )
    if "deepseek" in model_name.lower():
        return ChatDeepSeek(
            model=model_name,
            temperature=temperature,
        )
    if "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )
    if "qwen" in model_name.lower() or "kimi" in model_name.lower():
        return ChatTogether(
            # together_api_key="YOUR_API_KEY",
            temperature=temperature,
            model=model_name,
        )
    raise ValueError(f"Model {model_name} not found")
