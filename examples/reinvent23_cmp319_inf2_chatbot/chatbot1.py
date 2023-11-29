class APIIngress:
    def __init__(self, llama_model_handle):
        self.handle = llama_model_handle

    async def __call__(self):
        return "Hello re:Invent!\n"


# TODO: Create app via .bind()
