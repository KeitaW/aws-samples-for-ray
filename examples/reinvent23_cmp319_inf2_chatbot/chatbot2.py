from starlette.requests import Request
from ray import serve


@serve.deployment
class APIIngress:
    def __init__(self, llama_model_handle):
        self.handle = llama_model_handle
        self.handle = self.handle.options(use_new_handle_api=True)  #

    async def __call__(self, req: Request):
        sentence = req.query_params.get("sentence")  #
        # TODO:
        # Invoke infer() within LlamaModel deployment
        # Return result
        return sentence


@serve.deployment
class LlamaModel:
    def __init__(self):
        pass

    def infer(self, sentence):
        return "Hello from LlamaModel!\n"


app = APIIngress.bind()  # TODO: bind LlamaModel handle
