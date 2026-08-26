import os

import requests

from dotenv import load_dotenv


load_dotenv()


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.docker.internal:11434/api/generate"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemma3:1b"
)

class LLM:

    def __init__(
        self,
        model_name: str = MODEL_NAME
    ):

        self.model_name = model_name


    def generate(
        self,
        prompt: str
    ) -> str:

        response = requests.post(

            OLLAMA_URL,

            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },

            timeout=120
        )


        response.raise_for_status()


        data = response.json()


        return data["response"]


if __name__ == "__main__":

    llm = LLM()


    prompt = """
Explain what a Python function is
in two sentences.
"""


    answer = llm.generate(
        prompt
    )


    print("\nAnswer:\n")

    print(answer)