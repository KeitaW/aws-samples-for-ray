from fastapi import FastAPI
from ray import serve
import torch
from transformers import AutoTokenizer
import os

app = FastAPI()

hf_model = "NousResearch/Llama-2-7b-chat-hf"
local_model_path = f"{hf_model.replace('/','_')}-split"

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, llama_model_handle) -> None:
        self.handle = llama_model_handle

    @app.get("/infer")
    async def infer(self, sentence: str):
        ref = await self.handle.infer.remote(sentence)
        result = await ref
        return result

@serve.deployment(
    ray_actor_options={"resources": {"neuron_cores": 12},
                       "runtime_env": {"env_vars": {"NEURON_CC_FLAGS": "-O1"}}},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)

class LlamaModel:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer
        from transformers_neuronx.llama.model import LlamaForSampling
        from transformers import LlamaForCausalLM
        from transformers_neuronx.module import save_pretrained_split

        if not os.path.exists(local_model_path):
            print(f"Saving model split for {hf_model} to local path {local_model_path}")
            self.model = LlamaForCausalLM.from_pretrained(hf_model)
            save_pretrained_split(self.model, local_model_path)
        else:
            print(f"Using existing model split {local_model_path}")

        print(f"Loading and compiling model {local_model_path} for Neuron")
        self.neuron_model = LlamaForSampling.from_pretrained(local_model_path, batch_size=1, tp_degree=12, amp='f16')
        self.neuron_model.to_neuron()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def infer(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(input_ids, sequence_length=1024, top_k=50)
        return [self.tokenizer.decode(seq) for seq in generated_sequences]

entrypoint = APIIngress.bind(LlamaModel.bind())