"""Holds constants."""

DEFAULT_MODEL = "gpt-oss:20b" # "gpt-5.2"

# Prices in USD as of December 30, 2025 (https://openai.com/api/pricing/)
MODEL_TO_INPUT_PRICE_PER_TOKEN = {
    "gpt-3.5-turbo-0125": 0.5 / 10**6,
    "gpt-4o-2024-08-06": 2.5 / 10**6,
    "gpt-4o-2024-05-13": 5 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.15 / 10**6,
    "o1-mini-2024-09-12": 3 / 10**6,
    "gpt-oss:20b": 0,
    "qwen3:8b": 0,
    "gpt-5": 1.25 / 10**6,
    "gpt-5-mini": 0.25 / 10**6,
    "gpt-5-nano": 0.05 / 10**6,
    "gpt-5.2": 1.75 / 10**6,
    "gpt-5.2-pro": 21 / 10**6,
}

MODEL_TO_OUTPUT_PRICE_PER_TOKEN = {
    "gpt-3.5-turbo-0125": 1.5 / 10**6,
    "gpt-4o-2024-08-06": 10 / 10**6,
    "gpt-4o-2024-05-13": 15 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.6 / 10**6,
    "o1-mini-2024-09-12": 12 / 10**6,
    "gpt-oss:20b": 0,
    "qwen3:8b": 0,
    "gpt-5": 10 / 10**6,
    "gpt-5-mini": 2 / 10**6,
    "gpt-5-nano": 0.4 / 10**6,
    "gpt-5.2": 14 / 10**6,
    "gpt-5.2-pro": 168 / 10**6,
}

FINETUNING_MODEL_TO_INPUT_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": 3.75 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.3 / 10**6,
}

FINETUNING_MODEL_TO_OUTPUT_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": 15 / 10**6,
    "gpt-4o-mini-2024-07-18": 1.2 / 10**6,
}

FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": 25 / 10**6,
    "gpt-4o-mini-2024-07-18": 3 / 10**6,
}

DEFAULT_FINETUNING_EPOCHS = 4

CONSISTENT_TEMPERATURE = 0.2
CREATIVE_TEMPERATURE = 0.8

PUBMED_TOOL_NAME = "pubmed_search"
PUBMED_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": PUBMED_TOOL_NAME,
        "description": "Get abstracts or the full text of biomedical and life sciences articles from PubMed Central.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A PubMed Central (PMC) search expression. Supports PubMed/PMC boolean syntax and field tags. Use uppercase logical operators (AND, OR, NOT), parentheses to group complex conditions. DO NOT use double quotes. Common field tags include: [ti] (title), [ab] (abstract), [tiab] (title+abstract), [au] (author), [jour] (journal), [pmcid] (PMCID). PubMed-style range filters (e.g., publication date) are also supported. Examples:\\n- (breast cancer) AND (deep learning OR machine learning)\\n- gene editing AND CRISPR NOT review[Publication Type]\\n- deep learning[ti] AND cancer[tiab]\\n- (immunotherapy[ti]) AND (Nature[jour] OR Science[jour])\\n- (graph neural network) AND (Lee J[au] OR Kim S[au])\\n- (cancer) AND (immunotherapy) AND (2019/01/01[Date - Publication] : 2025/12/31[Date - Publication])\\n- pmcid:PMC* AND COVID-19\\nNote: URL encoding should be handled by the caller; provide a plain-text search string here."
                },
                "num_articles": {
                    "type": "integer",
                    "description": "The number of articles to return from the search query.",
                },
                "abstract_only": {
                    "type": "boolean",
                    "description": "Whether to return only the abstract of the articles.",
                },
            },
            "required": ["query", "num_articles"],
        },
    },
}
