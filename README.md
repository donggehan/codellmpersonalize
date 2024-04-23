# LLM-Personalize: Aligning LLM Planners with Human Preferences via Reinforced Self-Training for Housekeeping Robots

This repo contains the implementation of our paper (under submission to IROS 2024) [LLM-Personalize: Aligning LLM Planners with Human Preferences via Reinforced Self-Training for Housekeeping Robots](https://donggehan.github.io/projectllmpersonalize/).

![Demo Animation](images/iros_demo.gif)



## Installation

### Conda Installation
Clone this repo, and follow the installation instruction from the [Housekeep Benchmark](https://github.com/yashkant/housekeep).

The above installation is sufficient to run a human demonstration agent.
In addition, this work uses OpenAI GPT-3.5-Turbo. Therefore, to train and run an LLM-based agent using OpenAI GPT's, please setup an OpenAI account, and additionally install the python module:
``` 
pip install openai==0.27.9
```

### Docker
Alternatively, you can pull and use the docker: 
```
docker pull apple90gege/housekeepdocker:latest
```
And follow [installation guide](https://github.com/donggehan/habitatDockerBuilder/blob/master/installationGuides/housekeep_installation_guide_with_manual_habitat_docker.md) from [this repo](https://github.com/donggehan/habitatDockerBuilder).

### Troubleshoot headless rendering
This work runs the 3D habitat simulator on a headless GPU server. [Here](https://github.com/donggehan/eidf-epcc-cluster/blob/troubleshooting/troubleshooting-guides/headless_rendering_EGL_trouble_shoot.md) is an issue and its solution related to the installation. For more issues and troubleshoot guides please refer to the official [Habitat Simulator Repo](https://github.com/facebookresearch/habitat-sim).

## Running the code

#### Code Overview
- The model is mainly implemented in the folder ```cos_eor/policy```, i.e., ```context.py``` for the context generator, ```llm_planner.py``` for the LLM planner module, and ```hie_policy.py``` handles the coordination, communication and interactions among the context generator, the llm-planner and the low-level controller.
- The agent and simulation are configured in the ```cos_eor/configs/local/``` folder.
- The simulations are logged inside the ```logs/``` folder.
- The fine-tuning code are in the ```experiments/``` folder.

To create and run an agent, create the run configuration, then run a simulation as follows:
#### Step 1: Generating a configuration file
- Start by editing the configuration file ```cos_eor/configs/local/igib_v2_nav_sp.yaml```. 
  - To run the demonstrator agent, set the value of L212 to "manual" (```RL → POLICY → plan → single → llm_model → platform: "manual"```).
  - To run a zero-shot LLM agent with the OpenAI gpt-3.5-turbo-0613 model
    - set the value of L212 to "openai". (```RL → POLICY → plan → single → llm_model → platform: "openai"```). You may switch to a different GPT model at L243.
    - create a file ```cos_eor/configs/local/api_key.yaml``` with lines:
      ```
      key: "<YOUR_OPENAI_API_KEY>"
      organization: "<ORGANIZATION_ID>"
      ```
- Next, generate run configs by running the following command under the repo root.  
  ```
  ./generate_configs.sh test oracle Oracle llmzeroshot oracle
  ```
  This will create an experiment folder with name ```test``` under ```logs/test```. The configs will be stored in ```logs/test/configs```. After generating the configs, we're ready for step 2 to run a simulation.

#### Step 2: Running a simulation
- To run a simulation under the experiment ```test```, run the command 
  ```
  ./run_cli.sh test <ENV_ID> <NUM_EPISODES>
  ```
A list of environment ID ```<ENV_ID>``` are included in the file ```cos_eor/configs/local/envs.yaml```. For example, to run the agent in the environment ```pomaria_1_int``` for ```1``` episode, you can run the command as ```./run_cli.sh test pomaria_1_int 1```.

The simulation logs are then written to ```logs/test/demo/<ENV_ID>/data_<datetime>.json```. In addition, to record a video, you may change the config file L143 ```VIDEO_INTERVAL``` to 1 and find the video inside the log folder.

## Fine-tuning guide

### Demonstration Bootstrapping
- In this work, we first bootstrap the LLM with demonstrations by supervised fine-tuning via the OpenAI fine-tuning API. To annotate the demonstrations and upload the training data to fine-tune the LLM, run the notebook ```experiments/annotate_human_demo_pair.ipynb```. Note that for each fine-tuning job, the training data and fine-tuning job info can be found in the ```experiments/<FINETUNE_JOB_ID>``` folder.
- After the fine-tuning job has finished, collect the fine-tuned model ID from the OpenAI API. To run a new simulation with the fine-tuned agent, follow Step 1, change the model ID to the fine-tuned GPT model ID in ```cos_eor/configs/local/api_key.yaml```, generate the config and run a new simulation.


### Iterative Self-training
- After running the bootstrapped agent to collect simulated interactions, we can annotate the collected logs and fine-tune the LLM model again. This is similar to the demonstration bootstrapping step, and can be run with the notebook ```experiments/annotate_gpt3_pair.ipynb```.
- Finally, the notebook ```experiments/evaluate_gpt3_pair_in_domain.ipynb``` contains an example to evaluate the agents' rearrangement performance.


## Citing

Our paper is available on [Arxiv](https://arxiv.org/abs/2404.14285). If you find our code useful, please consider citing us!
```
@misc{han2024llmpersonalize,
      title={LLM-Personalize: Aligning LLM Planners with Human Preferences via Reinforced Self-Training for Housekeeping Robots}, 
      author={Dongge Han and Trevor McInroe and Adam Jelley and Stefano V. Albrecht and Peter Bell and Amos Storkey},
      year={2024},
      eprint={2404.14285},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```


## Acknowledgement
The [housekeep benchmark](https://yashkant.github.io/housekeep/) and the housekeep simulator code used in this work (included in this repo) is developed by Kant et. al. for their paper: [Housekeep: Tidying Virtual Households using Commonsense Reasoning.](https://arxiv.org/abs/2205.10712). The simulator is based on the [Habitat simulator](https://github.com/facebookresearch/habitat-sim) introduced in the paper [Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405).
