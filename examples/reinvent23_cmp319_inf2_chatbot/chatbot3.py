import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split
from starlette.requests import Request
from ray import serve


@serve.deployment
class APIIngress:
    def __init__(self, llama_model_handle):
        self.handle = llama_model_handle.options(use_new_handle_api=True)

    async def __call__(self, req: Request):
        sentence = req.query_params.get("sentence")
        res = self.handle.infer.remote(sentence)
        return await res


@serve.deployment(
    ray_actor_options={
        "resources": {"neuron_cores": 12},
        "runtime_env": {
            "env_vars": {
                "NEURON_CC_FLAGS": "-O1",
                "NEURON_COMPILE_CACHE_URL": "/home/ubuntu/neuron_demo/neuron-compile-cache",
            }
        },
    },
    num_replicas=1,
)
class LlamaModel:
    def __init__(self):
        hf_model = "NousResearch/Llama-2-7b-chat-hf"
        local_model_path = f"{hf_model.replace('/','_')}-split"

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = LlamaForCausalLM.from_pretrained(hf_model)

        # TODO:
        # Split model state_dict for fast loading onto Inferentia
        # Set up model configuration for Inferentia
        # Compile and load model onto accelerators

    def infer(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")

        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(input_ids, sequence_length=512, top_k=20)

        return "\n".join([self.tokenizer.decode(seq) for seq in generated_sequences])


app = APIIngress.bind(LlamaModel.bind())
