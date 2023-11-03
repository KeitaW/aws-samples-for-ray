import os
import torch
from ray import serve
from starlette.requests import Request
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

hf_model = "NousResearch/Llama-2-7b-chat-hf"
local_model_path = f"{hf_model.replace('/','_')}-split"


@serve.deployment(num_replicas=1)
class APIIngress:
    def __init__(self, llama_model_handle) -> None:
        self.handle = llama_model_handle

    async def __call__(self, req: Request):
        sentence = req.query_params.get("sentence")
        ref = await self.handle.infer.remote(sentence)
        result = await ref
        return result


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
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 1,
        "health_check_timeout_s": 600,
    },
)
class LlamaModel:
    def __init__(self):
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
            generated_sequences = self.neuron_model.sample(
                input_ids, sequence_length=512, top_k=20
            )
        return [self.tokenizer.decode(seq) for seq in generated_sequences]


app = APIIngress.bind(LlamaModel.bind())
