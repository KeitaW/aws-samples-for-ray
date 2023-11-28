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

        if not os.path.exists(local_model_path):
            print(f"Saving model split for {hf_model} to local path {local_model_path}")
            self.model = LlamaForCausalLM.from_pretrained(hf_model)
            save_pretrained_split(self.model, local_model_path)
        else:
            print(f"Using existing model split {local_model_path}")

        print(f"Loading and compiling model {local_model_path} for Neuron")
        self.neuron_model = LlamaForSampling.from_pretrained(
            local_model_path, batch_size=1, tp_degree=12, amp="f16"
        )
        self.neuron_model.to_neuron()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def infer(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(input_ids, sequence_length=512, top_k=20)
        return "\n".join([self.tokenizer.decode(seq) for seq in generated_sequences])


app = APIIngress.bind(LlamaModel.bind())
