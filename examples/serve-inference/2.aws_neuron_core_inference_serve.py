import os
import torch
from ray import serve
from starlette.requests import Request
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

hf_model = "NousResearch/Llama-2-7b-chat-hf"
local_model_path = f"/home/ubuntu/{hf_model.replace('/','_')}-split"


class APIIngress:
    def __init__(self, llama_model_handle) -> None:
        self.handle = llama_model_handle

    async def __call__(self, req: Request):
        sentence = req.query_params.get("sentence")
        ref = await self.handle.infer.remote(sentence)
        result = await ref
        return result


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
        print(f"compiling...")
        self.neuron_model.to_neuron()
        print(f"compiled!")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def infer(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(
                input_ids, sequence_length=512, top_k=20
            )
        return [self.tokenizer.decode(seq) for seq in generated_sequences]


app = APIIngress.bind(LlamaModel.bind())
