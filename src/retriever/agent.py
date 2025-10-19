from datasets import load_dataset
from retriever.llm_base import DEFAULT_TEMPERATURE, get_model_by_name
from typing import List, Type
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
)
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from retriever.search_provider import (
    SemanticScholarSearchProvider,
    PaperSearchResult,
    SemanticScholarWebSearchProvider,
)
from tempfile import NamedTemporaryFile
import requests
from PyPDF2 import PdfReader
from functools import reduce
from utils.str_matcher import is_similar
from utils.data_model import Output, OutputSearchOnly, OutputNoRead
import time
import tiktoken
import re
import json


ENCODING = tiktoken.encoding_for_model("gpt-4o")

def estimate_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def extract_last_json_block(text: str) -> str:
    """
    Extract the last top-level JSON object from a string using brace counting.
    Works even if earlier parts of the string contain stray '{' (e.g., LaTeX).
    """
    stack = []
    start_idx = None
    end_idx = None

    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            if not stack:
                end_idx = i
            stack.append('}')
        elif text[i] == '{':
            if stack:
                stack.pop()
                if not stack:
                    start_idx = i
                    break

    if start_idx is not None and end_idx is not None:
        json_str = text[start_idx:end_idx + 1]
        return json_str.strip()
    else:
        raise ValueError("No complete JSON block found in response.")

class BaseAgent:

    history: List[SystemMessage | AIMessage | HumanMessage] = []

    def get_history(self, ignore_system_messages=True):
        messages = self.history
        if ignore_system_messages:
            messages = [
                message
                for message in messages
                if not isinstance(message, SystemMessage)
            ]
        converted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                converted_messages.append(
                    {"role": "system", "content": message.content}
                )
            if isinstance(message, AIMessage):
                converted_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            if isinstance(message, HumanMessage):
                converted_messages.append({"role": "user", "content": message.content})
            if isinstance(message, ToolMessage):
                converted_messages.append({"role": "tool", "content": message.content})
        return converted_messages


class PaperNotFoundError(ValueError):
    pass


class LLMSelfAskAgentPydantic(BaseAgent):

    prompts = {
        "zero_shot_search": ["src/retriever/prompt_templates/zero_shot_search.txt", 0],
        "one_shot_search": ["src/retriever/prompt_templates/one_shot_search.txt", 0],
        "few_shot_search": ["src/retriever/prompt_templates/few_shot_search.txt", 1],
        "few_shot_tool": ["src/retriever/prompt_templates/few_shot_tool.txt", 1],
        "few_shot_search_no_read": [
            "src/retriever/prompt_templates/few_shot_search_no_read.txt",
            0,
        ],
    }
    human_intros = [
        "You are now given an excerpt. Find me the paper cited in the excerpt, using the tools described above.",
        "You are now given an excerpt. Find me the paper cited in the excerpt, using the tools described above. Please make sure that the paper you select really corresponds to the excerpt: there will be details mentioned in the excerpt that should appear in the paper. If you read an abstract and it seems like it could be the paper we’re looking for, read the paper to make sure. Also: sometimes you’ll read a paper that cites the paper we’re looking for. In such cases, please go to the references in order to find the full name of the paper we’re looking for, and search for it, and then select it.",
    ]

    def __init__(
        self,
        model_name: str,
        temperature=DEFAULT_TEMPERATURE,
        search_limit=10,
        only_open_access=True,
        use_web_search=False,
        prompt_name: str = "default",
        pydantic_object: Type[Output] | Type[OutputSearchOnly] = Output,
    ) -> None:
        self.prompt_template_path = self.prompts[prompt_name][0]
        self.human_intro = self.human_intros[self.prompts[prompt_name][1]]
        self.model = get_model_by_name(model_name, temperature=temperature)
        print("Using model:", self.model)
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
        if use_web_search:
            self.search_provider = SemanticScholarWebSearchProvider(
                limit=search_limit, only_open_access=only_open_access
            )
            self.search_provider.s2api.warmup()
        else:
            self.search_provider = SemanticScholarSearchProvider(
                limit=search_limit, only_open_access=only_open_access
            )
        self.source_papers_title: List[str] = []
        self.paper_dict = self._load_paper_dict()
        self.reset()
    def _load_paper_dict(self):
        dataset = load_dataset("KathCYM/CiteMEFull", split="train")  # or specify another split if needed
        paper_dict = {}

        for item in dataset:
            title = item.get("source_paper_title")
            text = item.get("source_paper_text")
            if title and text:
                paper_dict[title] = text

        return paper_dict

    def reset(self, source_papers_title: List[str] = []):
        with open(self.prompt_template_path, "r") as f:
            system_prompt = f.read()
        system_prompt = system_prompt.replace(
            "<FORMAT_INSTRUCTIONS>", self.parser.get_format_instructions()
        )
        if isinstance(self.model, ChatOpenAI) and self.model.model_name.startswith('o1'):
            self.history = [HumanMessage(content=system_prompt)]
        else:
            self.history = [SystemMessage(content=system_prompt)]
        self.paper_buffer: List[List[PaperSearchResult]] = []
        self.source_papers_title = source_papers_title

    def __find_paper_by_id(self, paper_id: str, in_entire_buffer=False):
        search_buffer = self.paper_buffer[-1] if len(self.paper_buffer) > 0 else []
        if in_entire_buffer and len(self.paper_buffer) > 1:
            search_buffer = reduce(
                lambda a, b: a + b, self.paper_buffer
            )  # Flatten the buffer
        for paper in search_buffer:
            if paper.paperId == paper_id:
                return paper
        raise PaperNotFoundError(
            f"Paper {paper_id} not found in buffer: {[p.paperId for p in search_buffer]}. Please try a different paper."
        )

    def _search_relevance(self, query: str, year: str):
        papers = self.search_provider(query, year)
        return self.__process_search(papers)

    def _search_citation_count(self, query: str, year: str):
        papers = self.search_provider.citation_count_search(query, year)
        return self.__process_search(papers)

    def _search_snippet(self, query:str, year: str, src_paper_title: str):
        results = self.search_provider.snippet_search(query, year, src_paper_title)
        print("search snippet results:", results)
        return HumanMessage(results)

    def __process_search(self, papers: List[PaperSearchResult]):
        filtered_papers: List[PaperSearchResult] = papers
        for paper in papers:
            for source_paper_title in self.source_papers_title:
                if is_similar(source_paper_title, paper.title):
                    filtered_papers.remove(paper)
        papers = filtered_papers

        self.paper_buffer.append(papers)
        papers_str = ""
        for paper in papers:
            papers_str += (
                f"- Paper ID: {paper.paperId}\n"
                + f"   Title: {paper.title}\n"
                + f"   Abstract: {paper.abstract}\n"
                + f"   Citation Count: {paper.citationCount}\n\n"
            )
        if len(papers) == 0:
            papers_str = "No papers were found for the given search query. Please use a different query."
        return HumanMessage(content=papers_str.strip())

    def find_sentences_with_substring(self, text, substring):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if substring.lower() in s.lower()]

    def _ask_for_more_context(self, query:str, paper_title:str):
        if paper_title not in self.paper_dict:
            return HumanMessage(content="Paper not found in database. Please use search_relevance action with the paper title to locate the paper.")
        paper_text = self.paper_dict[paper_title]
        clean_text = re.sub(r'\s+', ' ', paper_text).strip()
        pattern = re.escape(query).replace(r'\[CITATION\]', r'\[\d+\]')
        sentence_pattern = re.compile(r'(?<=[.?!])\s+(?=[A-Z])')
        sentences = sentence_pattern.split(clean_text)

        # Find the index of the sentence containing the match
        match_idx = -1
        for i, sentence in enumerate(sentences):
            if re.search(pattern, sentence):
                match_idx = i
                break

        if match_idx == -1:
            return HumanMessage(content="No matching context found. Please try with a different query.")

        # Extract window of sentences around the match
        start = max(0, match_idx - window)
        end = min(len(sentences), match_idx + window + 1)
        context = ' '.join(sentences[start:end]).strip()

        return HumanMessage(content=context)

    def _read_and_find_in_text(self, paper_id: str, query: str):
        text_msg = self._read(paper_id)
        text = text_msg.content

        # Simple heuristic: check for known error phrases
        if "error reading" in text.lower() or "does not have an open access" in text.lower():
            print("error reading", text)
            return text_msg

        matches = self.find_sentences_with_substring(text, query)
        if matches:
            print(matches[0])
            return HumanMessage(content="\n".join(matches))
        else:
            print("no match")
            return HumanMessage(content=f"No sentence found containing '{query}'.")

    def handle_aaai(self, url):
        referer_url = url.replace("/download/", "/view/").rsplit("/", 1)[0]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": referer_url,
            "Connection": "keep-alive",
        }
        return requests.get(url, headers=headers).content

    def _read(self, paper_id: str):
        paper = self.__find_paper_by_id(paper_id)
        if paper.openAccessPdf.url is None or paper.openAccessPdf.url == '':
            return HumanMessage(content="This paper does not have an open access PDF.")
        try:
            print(f"reading {paper.openAccessPdf.url}")
            pdf_text = ""
            with NamedTemporaryFile(mode="wb", suffix=".pdf") as f:
                if 'aaai' in paper.openAccessPdf.url:
                    pdf_bytes = self.handle_aaai(paper.openAccessPdf.url)
                else:
                    pdf_bytes = requests.get(paper.openAccessPdf.url).content
                f.write(pdf_bytes)
                f.flush()
                reader = PdfReader(f.name)
                for i, page in enumerate(reader.pages):
                    pdf_text += page.extract_text()
            if len(pdf_text) == 0:
                pdf_text = (
                    f"There was an error reading the PDF. Please try a different paper. {paper.openAccessPdf.url}"
                )

            return HumanMessage(content=pdf_text)
        except Exception as e:
            return HumanMessage(
                content=f"There was an error reading the PDF. Please try a different paper. {e}"
            )

    def _select(self, paper_id: str):
        paper = self.__find_paper_by_id(paper_id, in_entire_buffer=True)
        return paper

    def _ask_llm(self, message, last_action=False) -> Output:
        self.history.append(message)
        if last_action:
            self.history.append(
                HumanMessage(
                    content="Caution, you have reached the maximum number of actions. Please select a paper."
                )
            )
        prompt = ChatPromptTemplate.from_messages(self.history)
        pipeline = prompt | self.model
        MAX_RETRIES = 3
        RETRY_DELAY = 60
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print("hey")
                response: BaseMessage = pipeline.invoke({})
                cleaned = extract_last_json_block(response.content)
                self.history.append(response)
                
                input_tokens = sum(estimate_tokens(m.content) for m in self.history)
                output_tokens = estimate_tokens(response.content)
                total_tokens = input_tokens + output_tokens
                print('total_tokens updated:', total_tokens)

                if total_tokens > 25000:
                    print("⚠️ Near TPM limit, sleeping 60s...")
                    time.sleep(60)

                # Parse only the part starting from the first '{'
                parsed_response = AIMessage(content=cleaned.strip())
                return self.parser.invoke(parsed_response)

            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {e} \n{response}")
                if attempt < MAX_RETRIES:
                    print(f"🔁 Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("🚫 Max retries reached. Raising error.")
                    raise e

        """response: BaseMessage = pipeline.invoke({})
        self.history.append(response)
        
        # ignore input before first '{' to avoid parsing errors
        response = AIMessage(
            content=response.content[response.content.find("{") :].strip()
        )"""

    def get_paper_buffer(self):
        paper_buffer = []
        for buffer in self.paper_buffer:
            tmp_buffer = [paper.model_dump() for paper in buffer]
            paper_buffer.append(tmp_buffer)
        return paper_buffer

    def __call__(self, excerpt: str, year: str, src_paper_title: str, max_actions=5):
        prompt = f"The excerpt is from paper title '{self.source_papers_title}':\n"
        message = HumanMessage(content=self.human_intro + "\n\n" + f"{excerpt}")
        for i in range(max_actions):
            print("ask llm")
            response = self._ask_llm(message, last_action=(i == max_actions - 1))
            print("response", response)
            if response.action.name == "search_relevance":
                message = self._search_relevance(response.action.query, year)
            elif response.action.name == "search_citation_count":
                message = self._search_citation_count(response.action.query, year)
            elif response.action.name == "read":
                try:
                    message = self._read(response.action.paper_id)
                except PaperNotFoundError as e:
                    message = HumanMessage(content=f"{e}")
            elif response.action.name == "select":
                try:
                    return self._select(response.action.paper_id)
                except PaperNotFoundError as e:
                    message = HumanMessage(content=f"{e}")
            elif response.action.name == "find_in_text":
                try:
                    message = self._read_and_find_in_text(response.action.paper_id, response.action.query)
                except PaperNotFoundError as e:
                    message = HumanMessage(content=f"{e}")
            elif response.action.name == "ask_for_more_context":
                message = self._ask_for_more_context(response.action.query, response.action.paper_title)
            elif response.action.name == "search_text_snippet":
                message = self._search_snippet(response.action.query, year, src_paper_title)
            else:
                raise ValueError("Unknown action")

        raise ValueError("Max actions reached")


class LLMNoSearch(BaseAgent):

    prompts = {
        "zero_shot_no_search": "src/retriever/prompt_templates/zero_shot_no_search.txt",
        "few_shot_no_search": "src/retriever/prompt_templates/few_shot_no_search.txt",
    }

    def __init__(
        self,
        model_name: str,
        temperature=DEFAULT_TEMPERATURE,
        prompt_name: str = "zero_shot_no_search",
    ):
        self.model = get_model_by_name(model_name, temperature=temperature)
        self.paper_buffer: List[List[PaperSearchResult]] = []
        self.prompt_template_path = self.prompts[prompt_name]
        self.reset()

    def _ask_llm(self, message):
        self.history.append(message)
        response = self.model(self.history)
        self.history.append(response)
        return response

    def get_paper_buffer(self):
        return []

    def reset(self, source_papers_title: List[str] = []):
        with open(self.prompt_template_path, "r") as f:
            system_prompt = f.read()
        if isinstance(self.model, ChatOpenAI) and self.model.model_name.startswith('o1'):
            self.history = [HumanMessage(content=system_prompt)]
        else:
            self.history = [SystemMessage(content=system_prompt)]

    def __call__(self, excerpt: str, year: str = "", max_actions=5):
        message = HumanMessage(
            content="You are now given an excerpt. Find me the paper cited in the excerpt. Only return the paper title, nothing else.\n\n"
            + f"{excerpt}"
        )
        response = self._ask_llm(message)
        content = response.content.split(":")
        if (
            content[0].lower().startswith("based on")
            or "guess" in content[0]
            or "most likely" in content[0]
        ):
            content = ":".join(content[1:]).strip()
        else:
            content = response.content
        return PaperSearchResult(
            paperId=None,
            title=content,
            authors=[],
            abstract=None,
            venue=None,
            year=None,
            citationCount=None,
            openAccessPdf=None,
        )
