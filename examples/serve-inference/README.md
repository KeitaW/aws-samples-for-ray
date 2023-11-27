# Deploy Chat bot on Inferentia with Ray

This example compiles Open LLAMA-3B model and deploys the model on Inferentia (Inf2)  instance
using Ray Serve and FastAPI. It uses transformers-neuronx to shard the model across devices/neuron cores via Tensor parallelism. 


## Step1: Infrastructure Setup 

* Clone this repo to run the example on your local environment:

```bash
git clone https://github.com/aws-samples/aws-samples-for-ray
cd aws-samples-for-ray/examples/serve-inference
```

* Replace subnet and security-group where you intend to launch the cluster in `1.cluster-inference-serve.yaml`

```
sed -i 's/subnet-replace-me/subnet-ID/g' cluster-inference-serve.yaml
sed -i 's/sg-replace-me/sg-ID/g' cluster-inference-serve.yaml
```

1. Start your Ray cluster from your local laptop (pre-requisite of Ray installation):

    ```bash
    ray up 1.cluster-inference-serve.yaml
    ```

    You will be prompted to confirm the cluster creation as follows. Input `y` and proceed.

    ```console
    Cluster: inference-serve

    2023-11-27 15:37:17,639 INFO util.py:375 -- setting max workers for head node type to 0
    Loaded cached provider configuration
    If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
    AWS config
    IAM Profile: ray-autoscaler-v1 [default]
    EC2 Key pair (all available node types): ray-autoscaler_us-west-2 [default]
    VPC Subnets (all available node types): subnet-0a910e572266c13bd [default]
    EC2 Security groups (ray.head.default): sg-0581f8300b3b2455e [default]
    EC2 Security groups (ray.worker.default): sg-0294e801896f7b828
    EC2 AMI (all available node types): ami-0396c2a8448f872d2

    No head node found. Launching a new cluster. Confirm [y/N]: 
    ```

    After the command, we have 1 head node and 1 worker node. 

2. Log in to the head node.
    Once cluster is launched, you can login to the head node with the following command.

    ```bash
    ray attach 1.cluster-inference-serve.yaml
    ```

    You will see terminal of the head node as follows.

    ```console
    2023-11-27 15:24:55,387 INFO util.py:375 -- setting max workers for head node type to 0
    Loaded cached provider configuration
    If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
    Fetched IP: 44.242.165.202
    Warning: Permanently added '44.242.165.202' (ED25519) to the list of known hosts.
    ubuntu@ip-10-0-95-229:~$ 
    ```

    Once log into the node mode the current working directory to `neuron_demo`.

    ```bash
    cd ~/neuron_demo
    ```

    The rest of the contents assume that you are working on the node and directory. 

## Step 2: Deploy Llama2 with Ray

Now that we have a ray cluster with Inf2 instances, let's deploy Llama2 model on the infrastructure. `2.aws_neuron_core_inference_serve.py` contains basic ray serve setup for this part.

We can deploy `app` defined in the script as follows.






3. Deploy the model using ray serve with `serve` command.


```
ray exec cluster-inference-serve.yaml \
'source aws_neuron_venv_pytorch/bin/activate && cd neuron_demo && serve run aws_neuron_core_inference_serve:entrypoint --runtime-env-json="{\"env_vars\":{\"NEURON_CC_FLAGS\": \"--model-type=transformer-inference\",\"FI_EFA_FORK_SAFE\":\"1\"}}"' \
--tmux
```

* Wait for the serve deployment to complete (typically takes ~5minutes)
```
ray exec cluster-inference-serve.yaml 'source aws_neuron_venv_pytorch/bin/activate && serve status'
```
Sample expected output of serve status
```
proxies:
  proxy-uuid: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694558868.7500472
    deployments:
      LlamaModel:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      APIIngress:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```


## Usage
Attach to the head node of the Ray cluster
```
ray attach cluster-inference-serve.yaml
```

Navigate to the python interpreter on head node
```
source aws_neuron_venv_pytorch/bin/activate && python
```

Start using the model
```
import requests

response = requests.get(f"http://127.0.0.1:8000/infer?sentence=AWS is super cool")
print(response.status_code, response.json())
```

## Demo with gradio
The demo file gradio_ray_serve.py integrates Llama2 with Gradio app on Ray Serve. Llama 2 inference is deployed through Gradio app on Ray Serve so it can process and respond to HTTP requests.
```
source aws_neuron_venv_pytorch/bin/activate
pip install gradio
serve run gradio_ray_serve:app --runtime-env-json='{"env_vars":{"NEURON_CC_FLAGS": "--model-type=transformer-inference", "FI_EFA_FORK_SAFE":"1"}}'
``` 

## Teardown
To teardown the cluster/resources
```
ray down cluster-inference-serve.yaml -y
```


Generate file
On headnode
source /opt/aws_neuron_venv_pytorch/bin/activate && serve run neuron_demo/config.yaml

dashboard tunneling


## TMP

### How to run serve

```bash
# ray get-head-ip cluster-inference-serve.yaml 
# 
serve build aws_neuron_core_inference_serve:app -o config.yaml
ray up cluster-inference-serve.yaml
source /opt/aws_neuron_venv_pytorch/bin/activate
serve run config.yaml
curl http://127.0.0.1:8000/infer?sentence=AWS 
hey -z 10m -t 0 http://127.0.0.1:8000/infer?sentence=AWS
hey -c 1 -q 0.1 -n 3 http://127.0.0.1:8000/infer?sentence=AWS
hey -c 100 -q 1 -n 100 http://127.0.0.1:8000/infer?sentence=AWS
```
