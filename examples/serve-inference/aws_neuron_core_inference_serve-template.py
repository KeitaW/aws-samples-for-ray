import os
import torch
from ray import serve
from starlette.requests import Request
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

@serve.deployment(
)
class APIIngress:
    def __init__(self, llama_model_handle) -> None:
        self.handle = llama_model_handle

    async def __call__(self):
        return "hello world!"

app = APIIngress.bind()
