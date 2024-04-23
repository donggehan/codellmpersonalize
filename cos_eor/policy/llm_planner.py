# A class which prepares the llm models for use as planner
from abc import ABC, abstractmethod
import random
from cos_eor.utils.constants import *
import numpy as np
import torch
import openai
import re
import os
import time
import json

class PlanModule(ABC):
    def __init__(self, config, num_envs):
        self.config = config
        self.num_envs = num_envs
        self.fail_threshold = self.config.get(FAIL_THRESHOLD, 30)
        self.prompt_threshold = self.config.get(PROMPT_THRESHOLD, 150)
        self.allowed_options = {REC: True, OBJ: True, ROOM: False, PRIMITIVE: False}
        if self.config.get(OPTION,{}).get(ROOM, False):
            self.allowed_options[ROOM] = True
            # get a list of available rooms
        if self.config.get(OPTION, {}).get(PRIMITIVE, False):
            self.allowed_options[PRIMITIVE] = True
    
    def reset(self, key_translator=None, house_logger=None):
        self.key_translator = key_translator
        self.house_logger = house_logger
    
    # extract the executable action steps from the step
    def postprocess(self, step, observed):
        # We will keep available actions to "go to item", "pickup item", "put on receptacle"
        # extract the target item, target receptacle, and action
        # An action will be in the form of "go to obj/rec", "look  at obj/rec", "pick obj", "place obj"
        # Add primitive actions: move forward, turn left, turn right, look up, look down
        PATTERNS = {
            "place": r"place (.+) on (.+)",
            "pick": r"pick up (.+)",
            "nav": r"go to (.+)",
            "look": r"look at (.+)",
            "punctuation": r'[\"\'\;]'
        }
        try:
            step = step.lower()
            target_obj = None
            primitive_actions = {MOVE_FORWARD: 1, TURN_LEFT: 2, TURN_RIGHT: 3, LOOK_UP: 4, LOOK_DOWN: 5}
            if step in primitive_actions:
                action = NAV
                target_obj = primitive_actions[step]
                target_rec = None
                target_type = PRIMITIVE
                return action, target_obj, target_rec, target_type, SUC
            elif re.match(PATTERNS["nav"], step):
                action = NAV
                match = re.match(PATTERNS["nav"], step)
                target = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
                if not self.key_translator.is_room(target) and not target in observed:
                    # check if object or rec was observed before
                    raise Exception(f'object or rec {target} not observed before, should explore first before navigating.')
                        
            elif re.match(PATTERNS["look"], step):
                action = LOOK
                match = re.match(PATTERNS["look"], step)
                target = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
            elif re.match(PATTERNS["pick"], step):
                action = PP
                match = re.match(PATTERNS["pick"], step)
                target = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
            elif re.match(PATTERNS["place"], step):
                action = PP
                match = re.match(PATTERNS["place"], step)
                target_obj = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
                target = re.sub(PATTERNS["punctuation"], '', match.group(2).strip())
            else:
                raise Exception(f'action {step} not found, should be one of go to, look at, pick, place')

            translate_suc = True
            if self.key_translator.is_room(target):
                result = self.key_translator.translate(target, simplify=False)
                target = result[VAL]
                translate_suc = result[FLAG]
            elif self.key_translator.simplify_keys:
                result = self.key_translator.translate(target, simplify=False)
                target = result[VAL]
                translate_suc = result[FLAG]
                if target_obj is not None:
                    result_obj = self.key_translator.translate(target_obj, simplify=False)
                    target_obj = result_obj[VAL]
                    translate_suc = result_obj[FLAG] and translate_suc
            if "urdf" in target:
                target_type = REC
                target_rec = target
            else:
                target_type = OBJ
                target_obj = target
                target_rec = None
            if translate_suc:
                flag = SUC
            else:
                flag = FAIL
        except Exception as e:
            print ('caught postprocess exception', e)
            action = None
            target_obj = None
            target_rec = None
            target_type = None
            flag = FAIL
        return action, target_obj, target_rec, target_type, flag
    
    def log_prompt_and_response(self):
        # self.prompt directly for single, or self.prompt[PLANNER] and self.prompt[ADAPTER] for multiagent
        if self.type == SINGLE:
            low_level_prompt = self.prompt
            high_level_prompt = ""
            high_level_text = ""
            low_level_text = self.response
            high_level_step = ""
        elif self.type == ADAPTER:
            high_level_prompt = self.prompt[PLANNER]
            low_level_prompt = self.prompt[ADAPTER]
            high_level_text = self.response[PLANNER]
            low_level_text = self.response[ADAPTER]
            high_level_step = self.get_high_level_plan_step() # only implemented in adapter mode
        elif self.type == SAYPLAN:
            high_level_prompt = self.prompt[SEARCH]
            low_level_prompt = self.prompt[PLANNER]
            high_level_text = ''
            low_level_text = self.response
            high_level_step = ''
        elif self.type == SAYCAN:
            high_level_prompt = ''
            low_level_prompt = self.prompt[PLANNER]
            high_level_text = ''
            low_level_text = self.response
            high_level_step = ''
        self.house_logger.summarise_record()
        self.house_logger.create_record(high_level_prompt, low_level_prompt, high_level_text, low_level_text, high_level_step)
        print ('new record created')
    
# oracle planner
class DummyPlanModule(PlanModule):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs)
        self.name = self.config[NAME]
        
# v.0 llm planner
class LLMPlanModule(PlanModule):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs)
        self.type = SINGLE
        self.name = self.config[NAME]
        self.prompt_prefix = ""
        self.planner_include_prev_steps_msg = False
        self.plan_mode = self.config[SINGLE][MODE]
        self.max_steps_per_prompt = self.config[SINGLE].get(MAX_STEPS_PER_PROMPT, 10)
        self.llm_model = None
        self.prompt = None
        self.response = None
        self.prompt_history = []
        self.response_history = [] # raw responses
        self.plan = None
        self._setup_llm(device)
        
    def _setup_llm(self, device):
        os.environ[CUDA_VISIBLE_DEVICES] = self.config[SINGLE][LLM_MODEL][CUDA_VISIBLE_DEVICES]
        # Define the cache directory path relative to the current working directory
        cache_dir_path = os.path.join(os.getcwd(), "cache")
        # Check if the cache directory exists, if not, create it
        if not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path)     
        os.environ[TRANSFORMERS_CACHE] = cache_dir_path
        os.environ[HF_HOME] = cache_dir_path
        platform = self.config[SINGLE][LLM_MODEL][PLATFORM]
        self.prompt_prefix = self.config[SINGLE][LLM_MODEL][platform][PROMPT][PREFIX]
        self.planner_include_prev_steps_msg = self.plan_mode != ONCE and self.config[SINGLE].get(PREV_STEPS_MSG, False)
        if platform == HF:
            self.llm_model = LLMModelHf(self.config[SINGLE][LLM_MODEL][HF], device)
        elif platform == OPENAI:
            self.llm_model = LLMModelOpenai(self.config[SINGLE][LLM_MODEL][OPENAI], device)
        elif platform == LLAMA:
            self.llm_model = LLMModelLlama(self.config[SINGLE][LLM_MODEL][LLAMA], device)
        elif platform == REPLAY:
            self.llm_model = LLMModelReplay(self.config[SINGLE][LLM_MODEL][REPLAY], device)
        elif platform == MANUAL:
            self.llm_model = LLMModelManual(self.config[SINGLE][LLM_MODEL][MANUAL], device)
        else:
            print (f'llm setup failed, cannot find llm platform {self.config[SINGLE][PLATFORM]}')

    # call at each episode
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)
        if self.config[SINGLE][LLM_MODEL][PLATFORM] in [MANUAL, REPLAY]:
            self.llm_model.reset()
        self.prompt_history = []
        self.response_history = []
        self.response = None
        self.prompt = None
        self.plan = LLMPlan()
        self.cur_step = 0
            
    # the external tool for sending in the prompt and get the plan
    def prompt_and_plan(self, prompt: dict, info=None):
        self.prompt = prompt
        self.prompt[PREFIX] = self.prompt_prefix
        self._generate_plan(info=info)
        self.prompt_history.append(self.prompt)
        print ('generated plan')
        print (self.plan.print_plan())
        self.log_prompt_and_response()
        
    def get_plan(self):
        content = self.plan.get_content()
        return content
    
    def pop_next_step(self):
        next_step = self.plan.pop_next_step()
        return next_step
    
    def get_next_step_and_increase_counter(self, nl_context=None, info=None):
        step = self.plan.get_step(self.cur_step)
        if step is not None:
            self.cur_step += 1
            return step
        else:
            if self.plan_mode == ONCE:
                return None
            elif self.plan_mode in [STEP, MULTISTEP]:
                self.prompt_and_plan(nl_context, info=info)
                step = self.plan.get_step(self.cur_step)
                self.cur_step += 1
                return step
    
    def get_step(self, step_index):
        step = self.plan.get_step(step_index)
        return step
    
    # Internal plan generation functions
    # a generic planner, either plan autoregressive or once
    def _generate_plan(self, info=None):
        prompt_planner = self._build_prompt_for_planner(self.prompt)
        self.llm_model.generate_response(prompt_planner, info)
        self._process_llm_response_to_plan()
        return self.plan.get_content()
        
    def _build_prompt_for_planner(self, prompt):
        # Input: prompt for planner in the form {user: x, system: y, prefix: z}
        assert (PREV_STEPS_MSG in prompt[USER])
        # add the execution history to the high-level planner
        msg = ""
        if self.planner_include_prev_steps_msg != False:
            prev_steps = self.house_logger.get_prev_steps()
            msg = "\nExecuted steps: " + ','.join(prev_steps[LOW_LEVEL][self.planner_include_prev_steps_msg])
        result = {
            USER: prompt[USER].replace(PREV_STEPS_MSG, msg),
            SYSTEM: prompt[SYSTEM],
            PREFIX: self.prompt[PREFIX]
        }
        print ('--- Planner Prompt ---')
        for k in result:
            print (f"[{k}] {result[k]}")
        print ('----------------------')
        self.prompt = result
        return result
    
    def _process_llm_response_to_plan(self):
        top_nl_response = self.prompt_prefix + self.llm_model.get_top_nl_response()
        print ("--- raw llm response --- ")
        print (top_nl_response)
        print ('------------------------')
        # post-processing
        top_nl_response = top_nl_response.lower().replace(' :', ":")
        self.response = self.llm_model.get_top_nl_response().lower().replace(' :', ":")
        
        if self.cur_step > self.prompt_threshold:
            top_nl_response = self.prompt_prefix + MISSION_COMPLETE
            top_nl_response = top_nl_response.lower().replace(' :', ":")
            self.response = MISSION_COMPLETE
        # processing the top response as one or multiple steps
        pattern = r"\n|step \d+\:|step\d+\:"
        # Split the string based on the pattern
        steps = [s.strip().strip('.').strip(',') for s in re.split(pattern, top_nl_response)]
        steps = [s for s in steps if len(s)>0] # exclude null results
        if self.plan_mode == ONCE:
            self.plan.add_steps(steps)
        elif self.plan_mode == MULTISTEP:
            self.plan.add_steps(steps[:self.max_steps_per_prompt]) # avoid cutting off the last step
        elif self.plan_mode == STEP:
            self.plan.add_step(steps[0])
        else:
            print (f"mode {self.plan_mode} not implemented")
        
# v.1 llm planner with adapter added
class LLMPlanAdapterModule(PlanModule):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs)
        self.name = self.config[NAME]
        self.type = ADAPTER
        self.prompt_prefix_planner = "" # loaded in setup
        self.prompt_prefix_adapter = "" # loaded in setup
        self.prompt_prefix_planner_msg = "" # loaded in setup
        self.plan_mode_planner = self.config[HIGH_LEVEL][MODE]
        self.planner_include_prev_steps_msg = False
        self.plan_mode_adapter = self.config[ADAPTER][MODE]
        self.adapter_mode = self.config[ADAPTER][MODE]
        self.llm_model_planner = None
        self.llm_model_adapter = None
        self.prompt_planner = None
        self.prompt_adapter = None
        self.response = None
        self.prompt_history = []
        self.response_history = [] # raw responses
        self.plan_planner = None
        self.plan_adapter = None
        self.high_level_plan_step = None
        self._setup_llm(device)
        
    def _setup_llm(self, device):
        os.environ[CUDA_VISIBLE_DEVICES] = self.config[HIGH_LEVEL][LLM_MODEL][CUDA_VISIBLE_DEVICES]
        # Define the cache directory path relative to the current working directory
        cache_dir_path = os.path.join(os.getcwd(), CACHE)
        # Check if the cache directory exists, if not, create it
        if not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path)     
        os.environ[TRANSFORMERS_CACHE] = cache_dir_path
        os.environ[HF_HOME] = cache_dir_path
        # setup high-level planner
        planner_platform = self.config[HIGH_LEVEL][LLM_MODEL][PLATFORM]
        self.prompt_prefix_planner = self.config[HIGH_LEVEL][LLM_MODEL][planner_platform][PROMPT][PREFIX]
        self.planner_include_prev_steps_msg = self.config[HIGH_LEVEL].get(PREV_STEPS_MSG, False)
        if planner_platform == HF:
            self.llm_model_planner = LLMModelHf(self.config[HIGH_LEVEL][LLM_MODEL][HF], device)
        elif planner_platform == OPENAI:
            self.llm_model_planner = LLMModelOpenai(self.config[HIGH_LEVEL][LLM_MODEL][OPENAI], device)
        elif planner_platform == LLAMA:
            self.llm_model_planner = LLMModelLlama(self.config[HIGH_LEVEL][LLM_MODEL][LLAMA], device)
        elif planner_platform == MANUAL:
            self.llm_model_planner = LLMModelManual(self.config[HIGH_LEVEL][LLM_MODEL][MANUAL], device)
        elif planner_platform == REPLAY:
            self.llm_model_planner = LLMModelReplay(self.config[HIGH_LEVEL][LLM_MODEL][REPLAY], device)
        else:
            print (f'planner llm setup failed, cannot find llm platform {self.config[HIGH_LEVEL][PLATFORM]}')
        # setup mid-level adapter
        adapter_platform = self.config[ADAPTER][LLM_MODEL][PLATFORM]
        self.prompt_prefix_adapter = self.config[ADAPTER][LLM_MODEL][adapter_platform][PROMPT][PREFIX]
        self.prompt_prefix_planner_msg = self.config[ADAPTER][LLM_MODEL][adapter_platform][PROMPT][PREFIX_PLANNER_MSG]
        if adapter_platform == HF:
            self.llm_model_adapter = LLMModelHf(self.config[ADAPTER][LLM_MODEL][HF], device)
        elif adapter_platform == OPENAI:
            self.llm_model_adapter = LLMModelOpenai(self.config[ADAPTER][LLM_MODEL][OPENAI], device)
        elif adapter_platform == LLAMA:
            self.llm_model_adapter = LLMModelLlama(self.config[ADAPTER][LLM_MODEL][LLAMA], device)
        elif adapter_platform == MANUAL:
            self.llm_model_adapter = LLMModelManual(self.config[ADAPTER][LLM_MODEL][MANUAL], device)
        elif adapter_platform == REPLAY:
            self.llm_model_adapter = LLMModelReplay(self.config[ADAPTER][LLM_MODEL][REPLAY], device)
        else:
            print (f'adapter llm setup failed, cannot find llm platform {self.config[ADAPTER][PLATFORM]}')
        

    # call at each episode
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)
        self.prompt_planner = None
        self.prompt_adapter = None
        self.response = None
        self.prompt_history = []
        self.response_history = []
        self.plan_planner = LLMPlan()
        self.plan_adapter = LLMPlan()
        self.cur_step_planner = 0
        self.cur_step_adapter = 0
        self.high_level_plan_step = None
            
    # the external tool for sending in the prompt and get the plan
    def prompt_and_plan(self, prompt, info=None):
        # rewire when and how to receive prompt
        self.prompt = prompt
        if type(prompt) == dict:
            if self.prompt[PLANNER] is not None:
                self.prompt[PLANNER][PREFIX] = self.prompt_prefix_planner
            self.prompt[ADAPTER][PREFIX] = self.prompt_prefix_adapter
            self._generate_plan(info=info)
            print ('generated adapter plan')
            print (self.plan_adapter.print_plan())
        else:
            print (f"prompt should be a dictionary for planner and adapter")
        self.prompt_history.append(self.prompt)
        self.response_history.append(self.response)
        self.log_prompt_and_response()
        
    def get_high_level_plan_step(self):
        return self.high_level_plan_step
            
        
    def get_plan(self):
        content = self.plan_adapter.get_content()
        return content
    
    def get_next_step_and_increase_counter(self, nl_context):
        step = self.plan_adapter.get_step(self.cur_step_adapter)
        if step is not None:
            self.cur_step_adapter += 1
            return step
        else:
            # prompt the high-level planner for the next step and let adapter expand
            self.prompt_and_plan(nl_context)
            step = self.plan_adapter.get_step(self.cur_step_adapter)
            self.cur_step_adapter += 1
            return step
            

    # internal use: extract high-level planner steps
    def _get_high_level_plan_next_step_and_increase_counter(self):
        step = self.plan_planner.get_step(self.cur_step_planner)
        self.cur_step_planner += 1
        return step
        
    def get_step(self, step_index):
        step = self.plan_planner.get_step(step_index)
        return step
    
    # Internal plan generation functions
    # first generate the next step by the high-level planner
    # then get the step to the mid-level adapter to expand into steps to be executed
    def _generate_plan(self, info=None):
        self.high_level_plan_step = None
        self.response = {}
        if self.prompt[PLANNER] is not None:
            prompt_planner = self._build_prompt_for_planner(self.prompt[PLANNER])
            self.llm_model_planner.generate_response(prompt_planner, info)
            self._process_llm_planner_response_to_plan()
            self.high_level_plan_step = self._get_high_level_plan_next_step_and_increase_counter()
        if MISSION_COMPLETE in self.high_level_plan_step.strip().lower():
            # Directly return mission complete
            self.plan_adapter.add_step(MISSION_COMPLETE)
            self.response[ADAPTER] = MISSION_COMPLETE
        else:
            if self.prompt[ADAPTER] is not None:
                prompt_adapter = self._build_prompt_for_adapter(self.prompt[ADAPTER], self.high_level_plan_step)
                self.llm_model_adapter.generate_response(prompt_adapter, info)
                self._process_llm_adapter_response_to_plan()
        return self.plan_adapter.get_content()
    
    def _build_prompt_for_planner(self, prompt):
        # Input: prompt for planner in the form {user: x, system: y, prefix: z}
        assert (PREV_STEPS_MSG in prompt[USER])
        # add the execution history to the high-level planner
        msg = ""
        if self.planner_include_prev_steps_msg != False:
            prev_steps = self.house_logger.get_prev_steps()
            msg = "\nExecuted steps: " + ','.join(prev_steps[HIGH_LEVEL][self.planner_include_prev_steps_msg])
        result = {
            USER: prompt[USER].replace(PREV_STEPS_MSG, msg),
            SYSTEM: prompt[SYSTEM],
            PREFIX: self.prompt[PLANNER][PREFIX]
        }
        print ('--- Planner Prompt ---')
        for k in result:
            print (f"[{k}] {result[k]}")
        print ('----------------------')
        self.prompt[PLANNER] = result
        return result
    
    def _build_prompt_for_adapter(self, prompt, high_level_plan_step): 
        # Input: prompt for adapter in the form {user: x, system: y, prefix: z}
        assert (MSG_PLANNER in prompt[USER])
        # add the msg from the high-level planner to the adapter
        msg = self.prompt_prefix_planner_msg + high_level_plan_step
        result = {
            USER: prompt[USER].replace(MSG_PLANNER, msg),
            SYSTEM: prompt[SYSTEM],
            PREFIX: self.prompt[ADAPTER][PREFIX]
        }
        print ('--- Adapter Prompt ---')
        for k in result:
            print (f"[{k}] {result[k]}")
        print ('----------------------')
        self.prompt[ADAPTER] = result
        return result
        
    def _process_llm_planner_response_to_plan(self):
        top_nl_response = self.prompt_prefix_planner + self.llm_model_planner.get_top_nl_response()
        print ("--- raw llm response --- ")
        print (top_nl_response)
        print ('------------------------')
        # post-processing
        top_nl_response = top_nl_response.lower().replace(' :', ":")
        self.response[PLANNER] = self.llm_model_planner.get_top_nl_response().lower().replace(' :', ":")
        if self.cur_step_planner > self.prompt_threshold:
            top_nl_response = self.prompt_prefix_planner + MISSION_COMPLETE
            self.response[PLANNER] = MISSION_COMPLETE
        # processing the top response as one or multiple steps
        pattern = r"\n|step \d+\:|step\d+\:"
        # Split the string based on the pattern
        steps = [s.strip().strip('.').strip(',') for s in re.split(pattern, top_nl_response)]
        steps = [s for s in steps if len(s)>0] # exclude null results
        if self.plan_mode_planner == ONCE:
            self.plan_planner.add_steps(steps)
        elif self.plan_mode_planner == STEP:
            self.plan_planner.add_step(steps[0])
        else:
            print (f"Planner mode {self.plan_mode} not implemented")
    
    def _process_llm_adapter_response_to_plan(self):
        top_nl_response = self.prompt_prefix_adapter + self.llm_model_adapter.get_top_nl_response()
        print ("--- raw llm response --- ")
        print (top_nl_response)
        print ('------------------------')
        # post-processing
        top_nl_response = top_nl_response.lower().replace(' :', ":")
        # processing the top response as one or multiple steps
        pattern = r"\n|step \d+\:|step\d+\:"
        # Split the string based on the pattern
        steps = [s.strip().strip('.').strip(',') for s in re.split(pattern, top_nl_response)]
        steps = [s for s in steps if len(s)>0] # exclude null results
        if self.plan_mode_adapter == ONCE:
            self.plan_adapter.add_steps(steps)
        elif self.plan_mode_adapter == STEP:
            self.plan_adapter.add_step(steps[0])
        else:
            print (f"Adapter mode {self.plan_mode_adapter} not implemented")
        self.response[ADAPTER] = self.llm_model_adapter.get_top_nl_response().lower().replace(' :', ":")
            
            
# baseline: SayPlan
class LLMSayPlanModule(PlanModule):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs)
        self.type = SAYPLAN
        self.name = self.config[NAME]
        self.prompt_prefix = ""
        self.planner_include_prev_steps_msg = False
        self.plan_mode = self.config[SAYPLAN][MODE]
        self.max_steps_per_prompt = self.config[SAYPLAN].get(MAX_STEPS_PER_PROMPT, 10)
        self.max_graph_command_steps = self.config[SAYPLAN].get(MAX_GRAPH_COMMAND_STEPS, 10)
        self.max_plan_retries_feedback = self.config[SAYPLAN].get(MAX_PLAN_RETRIES_FEEDBACK, 10)
        self.llm_model = None
        self.prompt = None
        self.prompt_search = None
        self.response = None
        self.response_search = None
        self.prompt_history = []
        self.response_history = [] # raw responses
        self.plan = None
        self._setup_llm(device)
        
    def _setup_llm(self, device):
        os.environ[CUDA_VISIBLE_DEVICES] = self.config[SAYPLAN][LLM_MODEL][CUDA_VISIBLE_DEVICES]
        # Define the cache directory path relative to the current working directory
        cache_dir_path = os.path.join(os.getcwd(), "cache")
        # Check if the cache directory exists, if not, create it
        if not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path)     
        os.environ[TRANSFORMERS_CACHE] = cache_dir_path
        os.environ[HF_HOME] = cache_dir_path
        platform = self.config[SAYPLAN][LLM_MODEL][PLATFORM]
        self.prompt_prefix = self.config[SAYPLAN][LLM_MODEL][platform][PROMPT][PREFIX]
        self.planner_include_prev_steps_msg = self.plan_mode != ONCE and self.config[SAYPLAN].get(PREV_STEPS_MSG, False)
        if platform == HF:
            self.llm_model = LLMModelHf(self.config[SAYPLAN][LLM_MODEL][HF], device)
        elif platform == OPENAI:
            self.llm_model = LLMModelOpenai(self.config[SAYPLAN][LLM_MODEL][OPENAI], device)
        elif platform == LLAMA:
            self.llm_model = LLMModelLlama(self.config[SAYPLAN][LLM_MODEL][LLAMA], device)
        elif platform == MANUAL:
            self.llm_model = LLMModelManual(self.config[SAYPLAN][LLM_MODEL][MANUAL], device)
        elif platform == REPLAY:
            self.llm_model = LLMModelReplay(self.config[SAYPLAN][LLM_MODEL][REPLAY], device)
        else:
            print (f'llm setup failed, cannot find llm platform {self.config[SAYPLAN][PLATFORM]}')

    # call at each episode
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)
        if self.config[SAYPLAN][LLM_MODEL][PLATFORM] in [MANUAL, REPLAY]:
            self.llm_model.reset()
        self.prompt_history = []
        self.response_history = []
        self.response = None
        self.prompt = None
        self.plan = LLMPlan()
        self.cur_step = 0
            
    # the external tool for sending in the prompt and get the plan
    def prompt_and_plan(self, prompt: dict, info=None):
        self.prompt = prompt
        if self.prompt[PLANNER] is not None:
            self.prompt[PLANNER][PREFIX] = self.prompt_prefix
        self._generate_plan(info=info)
        self.prompt_history.append(self.prompt)
        print ('generated plan')
        print (self.plan.print_plan())
        self.log_prompt_and_response()
        
    def get_plan(self):
        content = self.plan.get_content()
        return content
    
    def pop_next_step(self):
        next_step = self.plan.pop_next_step()
        return next_step
    
    def get_next_step_and_increase_counter(self, nl_context=None, info=None):
        step = self.plan.get_step(self.cur_step)
        if step is not None:
            self.cur_step += 1
            return step
        else:
            if self.plan_mode == ONCE:
                return None
            elif self.plan_mode in [STEP, MULTISTEP]:
                self.prompt_and_plan(nl_context, info=info)
                step = self.plan.get_step(self.cur_step)
                self.cur_step += 1
                return step
    
    def get_step(self, step_index):
        step = self.plan.get_step(step_index)
        return step
    
    # Internal plan generation functions
    # a generic planner, either plan autoregressive or once
    def _generate_plan(self, info=None):
        # first build prompt for semantic search
        # the info contains the house graph
        memory_expanded_nodes = {} # this will be used to decide the representation of the graph
        rooms = info[ROOM]
        graph = info[GRAPH].graph
        expanded_nodes = {r: False for r in rooms} # this keeps track of expanded nodes, close the node if matching contract
        for ic in range(self.max_graph_command_steps):
            print ('semantic search loop', ic, 'of', self.max_graph_command_steps)
            prompt_search = self._build_prompt_for_search(self.prompt[SEARCH], graph, memory_expanded_nodes, expanded_nodes)
            self.llm_model.generate_response(prompt_search, info)
            flag_parse_success, action, target = self._process_search_llm_response_to_command(info)
            if flag_parse_success:
                # update graph representation and send for prompting again
                if action == DONE:
                    # now we're ready for planning
                    break
                elif action == EXPAND:
                    if target in expanded_nodes:
                        expanded_nodes[target] = True
                    memory_expanded_nodes[target] = True
                elif action == CONTRACT:
                    if target in expanded_nodes:
                        expanded_nodes[target] = False
            else:
                # do not update graph and send for prompting again
                print ('parse failed, retrying...')
                continue
        
        replanning_history = []
        feedback_msg_history = []
        for ip in range(self.max_plan_retries_feedback):
            print ('iterative replanning loop', ip, 'of', self.max_plan_retries_feedback)
            feedback_msg = self._format_replanning_feedback_msg(replanning_history, feedback_msg_history)
            # then use the graph search results to prompt the planner
            prompt_planner = self._build_prompt_for_planner(self.prompt[PLANNER], graph, expanded_nodes, feedback_msg)
            self.llm_model.generate_response(prompt_planner, info)
            steps = self._process_llm_response_to_plan(dryrun=True)
            flag_verification_success, feedback_msg_verification = self.verify_plan_steps(info[GRAPH], steps)
            if flag_verification_success:
                selected_plan_steps = steps
                break
            else:
                # next replanning iteration
                feedback_msg_history.append(feedback_msg_verification)
                replanning_history.append(steps)
        if not flag_verification_success:
            # we'll use results from the last replanning iteration if re-planning threshold pass
            selected_plan_steps = steps
        self.add_plan_steps_to_plan(selected_plan_steps)
        return self.plan.get_content()
    
    def _format_replanning_feedback_msg(self, replanning_history, feedback_msg_history):
        feedback_msg = ''
        if len(feedback_msg_history) == 0:
            return feedback_msg
        else:
            proposed_steps = ', '. join(replanning_history[-1])
            feedback_msg = f'Feedback on proposed plan: {proposed_steps}: {feedback_msg_history[-1]}, please replan all steps.'
            return feedback_msg
    
    def verify_plan_steps(self, graph, plan_steps):
        feedback_msg = ''
        flag_success = True
        rooms, objects, recs, gripped_obj = self.get_rooms_objects_recs(graph)
        simulated_occupied_recs = {}
        for obj in objects:
            rec = objects[obj]
            simulated_occupied_recs[rec] = True
        simulated_agent_gripped_obj = gripped_obj
        
        for step in plan_steps:
            if 'mission complete' in step:
                continue 
            flag_parse_success, msg, action, target, target_obj = self.parse_step(step, rooms, objects, recs)
            if not flag_parse_success:
                # a debugging feedback
                # feedback_msg = msg
                # parsing error feedback
                feedback_msg = f'step {step} cannot be parsed, please replan all steps'
                flag_success = False
                return flag_success, feedback_msg
                
            # check action, if it is go to and look at then pass, if it is pick up, check agent's holding, if it is placement, check occupancy
            if action in ['go to', 'look']:
                # successful
                continue
            elif action == 'pick':
                if simulated_agent_gripped_obj in [NOTHING, '', None]:
                    # successful
                    simulated_agent_gripped_obj = target
                    if target in objects:
                        rec = objects[target]
                        simulated_occupied_recs[rec] = False
                else:
                    flag_success = False
                    feedback_msg = f'cannot pick up {target} as robot holding something.'
            elif action == 'place':
                if target in simulated_occupied_recs and simulated_occupied_recs[target]:
                    flag_success = False
                    feedback_msg = f'cannot place on {target} as receptacle occupied.'
                else:
                    if simulated_agent_gripped_obj not in [NOTHING, '', None]:
                        # successful
                        simulated_occupied_recs[target] = True 
                        simulated_agent_gripped_obj = NOTHING
        return flag_success, feedback_msg
    
    def get_rooms_objects_recs(self, graph):
        gripped_obj = graph.gripped_obj
        rooms = list(graph.graph.keys())
        objects = {}
        recs = []
        for room in graph.graph:
            recs.extend(graph.graph[room][REC])
            for obj in graph.graph[room][OBJ]:
                # Object and its rec
                objects[obj] = graph.graph[room][OBJ][obj][REC]
        return rooms, objects, recs, gripped_obj
    
    def parse_step(self, step, rooms, objects, recs):
        PATTERNS = {
            "place": r"place (.+) on (.+)",
            "pick": r"pick up (.+)",
            "nav": r"go to (.+)",
            "look": r"look at (.+)",
            "punctuation": r'[\"\'\;]'
        }
        action = None
        target_obj = None
        target = None
        try:
            step = step.lower()
            msg = ''
            if re.match(PATTERNS["nav"], step):
                action = 'go to'
                match = re.match(PATTERNS["nav"], step)
                target = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
                if not (target in rooms or target in objects or target in recs):
                    # check if object or rec was observed before
                    flag_parse_success=False
                    msg = f'{target} not found'
                else:
                    flag_parse_success = True
            elif re.match(PATTERNS["look"], step):
                action = 'look'
                match = re.match(PATTERNS["look"], step)
                target = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
                flag_parse_success = True
            elif re.match(PATTERNS["pick"], step):
                action = 'pick'
                match = re.match(PATTERNS["pick"], step)
                target = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
                if not target in objects:
                    flag_parse_success=False
                    msg = f'object {target} not found'
                else:
                    flag_parse_success = True
            elif re.match(PATTERNS["place"], step):
                action = 'place'
                match = re.match(PATTERNS["place"], step)
                target_obj = re.sub(PATTERNS["punctuation"], '', match.group(1).strip())
                target = re.sub(PATTERNS["punctuation"], '', match.group(2).strip())
                flag_parse_success = True
                if target_obj not in objects:
                    flag_parse_success=False
                    msg = f'object {target} not found'
                if target not in recs:
                    flag_parse_success=False
                    msg = f'receptacle {target} not found'
            else:
                # can't match action
                flag_parse_success=False
                msg = f'step {step} not correct'
        except Exception as e:
            msg = f'step {step} not correct'
            flag_parse_success = False
        return flag_parse_success, msg, action, target, target_obj
    

    def get_nl_subgraph(self, graph, expanded_nodes):
        # returns the view of subgraph with expanded rooms only
        # Example graph = {
        #     "Room1": {
        #         "receptacles": ["Receptacle1", "Receptacle2", "Receptacle3"],
        #         "objects": {
        #             "Object1": {"receptacle": "Receptacle1"},
        #             "Object2": {"receptacle": "Receptacle2"}
        #         }
        #     },
        #     "Room2": {
        #         "receptacles": ["Receptacle4", "Receptacle5"],
        #         "objects": {
        #             "Object3": {"receptacle": "Receptacle4"}
        #         }
        #     },
        # }
        
        # {nodes: {
            # [v] room: [{id: bobs_room}, {id: toms_room}, {id: jacks_room}, {id: kitchen}, {id: livingroom}], 
            # [x] pose: [{id: pose1}, {id: pose2}, {id: pose3}, {id: pose4}, {id: pose5}], 
            # [x] agent: [{location: bobs_room, id: agent}], 
            # asset: [
                # {room: toms_room, state: free, affordances: [release], id: bed2}, 
                # {room: toms_room, state: closed, affordances: [open, close, release], id: wardrobe2}, 
                # {room: kitchen, state: closed, affordances: [open, close, release], id: fridge},
                # {room: kitchen, affordances: [turn_on, turn_off], state: off, id: coffee_machine}, 
                # {room: bobs_room, state: free, affordances: [release], id: bed1}, 
                # {room: bobs_room, state: closed, affordances: [open, close, release], id: wardrobe1}], 
            # object: [{affordances: [pickup], state: inside_of(wardrobe1), attributes: "blue", id: coffee_mug}]}, 
            # links: [bobs_room↔pose1, bobs_room↔agent, bobs_room↔bed1, bobs_room↔wardrobe1, toms_room↔pose1, toms_room↔pose2, toms_room↔pose5, toms_room↔bed2, toms_room↔wardrobe2, jacks_room↔pose2, jacks_room↔pose3, kitchen↔pose3, kitchen↔pose4, kitchen↔pose5, kitchen↔fridge, kitchen↔coffee_machine, livingroom↔pose4, wardrobe1↔coffee_mug]
        # }
        descriptions = []
        # first add in the description of all rooms available
        rooms_description = ''
        rooms_list = list(graph.keys())
        nl_rooms = ', '.join(rooms_list)
        rooms_description = f'room: [{nl_rooms}]'
        descriptions.append(rooms_description)
        # now add in expanded room info for receptacles and objects
        recs_list = []
        objs_list = []
        occupied_recs = {}
        for room, data in graph.items():
            if room not in expanded_nodes:
                continue
            # now this room is expanded
            if len(data[REC]) > 0:
                recs_list.extend(data[REC])

            for obj, receptacle in data[OBJ].items():
                objs_list.append((obj, receptacle[REC]))
                occupied_recs[receptacle[REC]] = obj
                
        rec_states = {r: 'free' if r not in occupied_recs else 'occupied' for r in recs_list}

        # format receptacle list
        recs = ', '.join([f'{{id: {r}, state: {rec_states[r]}, affordances: [look at]}}' for r in recs_list])
        nl_recs = f'receptacles: [{recs}]'
        descriptions.append(nl_recs)
        # format object list
        objs = ', '.join([f'{{id: {obj[0]}, state: on {obj[1]}, affordances: [look at, pick up]}}' for obj in objs_list])
        nl_objs = f'objects: [{objs}]'
        descriptions.append(nl_objs)
        nl_graph = ", ".join(descriptions)
        return nl_graph
        
    def get_nl_memory_expanded_nodes(self, memory_expanded_nodes):
        # get description of previously explored nodes
        expanded_node_list = [n for n in memory_expanded_nodes if memory_expanded_nodes[n]]
        nl_memory = f'\nPreviously expanded nodes: {expanded_node_list}'
        return nl_memory

    def _build_prompt_for_search(self, prompt, graph, memory_expanded_nodes, expanded_nodes):
        # build the prompt for semantic search module
        msg_graph = self.get_nl_subgraph(graph, expanded_nodes)
        msg_memory = self.get_nl_memory_expanded_nodes(memory_expanded_nodes)
        result = {
            USER: prompt[USER].replace(GRAPH_PLACEHOLDER, msg_graph).replace(MEMORY_PLACEHOLDER, msg_memory),
            SYSTEM: prompt[SYSTEM],
            PREFIX: ""
        }
        # here we start with a contracted graph of rooms
        # 
        print ('--- Search Prompt ---')
        for k in result:
            print (f"[{k}] {result[k]}")
        print ('----------------------')
        return result
        
        
    def _build_prompt_for_planner(self, prompt, graph, expanded_nodes, msg_feedback=''):
        # Input: prompt for planner in the form {user: x, system: y, prefix: z}
        assert (PREV_STEPS_MSG in prompt[USER])
        # add the execution history to the high-level planner
        msg = ""
        msg_graph = self.get_nl_subgraph(graph, expanded_nodes)
        if self.planner_include_prev_steps_msg != False:
            prev_steps = self.house_logger.get_prev_steps()
            msg = "\nExecuted steps: " + ','.join(prev_steps[LOW_LEVEL][self.planner_include_prev_steps_msg])
        result = {
            USER: prompt[USER].replace(PREV_STEPS_MSG, msg).replace(GRAPH_PLACEHOLDER, msg_graph).replace(FEEDBACK_PLACEHOLDER, msg_feedback),
            SYSTEM: prompt[SYSTEM],
            PREFIX: self.prompt_prefix
        }
        print ('--- Planner Prompt ---')
        for k in result:
            print (f"[{k}] {result[k]}")
        print ('----------------------')
        return result
    
    def _process_llm_response_to_plan(self, dryrun=False):
        top_nl_response = self.prompt_prefix + self.llm_model.get_top_nl_response()
        print ("--- raw llm response --- ")
        print (top_nl_response)
        print ('------------------------')
        # post-processing
        top_nl_response = top_nl_response.lower().replace(' :', ":")
        self.response = self.llm_model.get_top_nl_response().lower().replace(' :', ":")
        
        if self.cur_step > self.prompt_threshold:
            top_nl_response = self.prompt_prefix + MISSION_COMPLETE
            top_nl_response = top_nl_response.lower().replace(' :', ":")
            self.response = MISSION_COMPLETE
        # processing the top response as one or multiple steps
        pattern = r"\n|step \d+\:|step\d+\:"
        # Split the string based on the pattern
        steps = [s.strip().strip('.').strip(',') for s in re.split(pattern, top_nl_response)]
        steps = [s for s in steps if len(s)>0] # exclude null results
        selected_steps = []
        if self.plan_mode == ONCE:
            selected_steps = steps[:]
        elif self.plan_mode == MULTISTEP:
            selected_steps = steps[:self.max_steps_per_prompt] # avoid cutting off the last step
        elif self.plan_mode == STEP:
            selected_steps = steps[:1]
        else:
            print (f"mode {self.plan_mode} not implemented")
        if not dryrun:
            self.add_plan_steps_to_plan(selected_steps)
        return selected_steps
                
    def add_plan_steps_to_plan(self, steps):
        self.plan.add_steps(steps)
            
    def _process_search_llm_response_to_command(self, info):
        rooms = info[ROOM]
        top_nl_response = self.llm_model.get_top_nl_response()
        print ("--- raw SEARCH llm response --- ")
        print (top_nl_response)
        print ('------------------------')
        # post-processing
        top_nl_response = top_nl_response.lower().replace(' :', ":")
        self.search_response = self.llm_model.get_top_nl_response().lower().replace(' :', ":")
        # processing into one of the three types of responses
        # 1. expand(room x)
        # 2. contract(room x)
        # 3. complete
        flag_parse_success = False
        words = self.search_response.split()
        action = ""
        target = ""
        if self.search_response == DONE:
            flag_parse_success = True
            action = DONE
        # Check for EXPAND or CONTRACT pattern
        elif len(words) >= 2 and words[0] in [EXPAND, CONTRACT]:
            action = words[0]
            target = " ".join(words[1:])
            if target in rooms:
                flag_parse_success = True
            else:
                flag_parse_success = False
        return flag_parse_success, action, target



# baseline: SayCan
class LLMSayCanModule(PlanModule):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs)
        self.type = SAYCAN
        self.name = self.config[NAME]
        self.prompt_prefix = ""
        self.planner_include_prev_steps_msg = False
        self.plan_mode = self.config[SAYCAN][MODE]
        self.max_steps_per_prompt = self.config[SAYCAN].get(MAX_STEPS_PER_PROMPT, 10)
        self.max_prompt_budget = self.config[SAYCAN].get(MAX_PROMPT_BUDGET, 100)
        self.llm_model = None
        self.prompt = None
        self.response = None
        self.prompt_history = []
        self.response_history = [] # raw responses
        self.plan = None
        self._setup_llm(device)
        
    def _setup_llm(self, device):
        os.environ[CUDA_VISIBLE_DEVICES] = self.config[SAYCAN][LLM_MODEL][CUDA_VISIBLE_DEVICES]
        # Define the cache directory path relative to the current working directory
        cache_dir_path = os.path.join(os.getcwd(), "cache")
        # Check if the cache directory exists, if not, create it
        if not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path)     
        os.environ[TRANSFORMERS_CACHE] = cache_dir_path
        os.environ[HF_HOME] = cache_dir_path
        platform = self.config[SAYCAN][LLM_MODEL][PLATFORM]
        self.prompt_prefix = self.config[SAYCAN][LLM_MODEL][platform][PROMPT][PREFIX]
        self.planner_include_prev_steps_msg = self.plan_mode != ONCE and self.config[SAYCAN].get(PREV_STEPS_MSG, False)
        if platform == HF:
            self.llm_model = LLMModelHf(self.config[SAYCAN][LLM_MODEL][HF], device)
        elif platform == OPENAI:
            self.llm_model = LLMModelOpenai(self.config[SAYCAN][LLM_MODEL][OPENAI], device)
        elif platform == LLAMA:
            self.llm_model = LLMModelLlama(self.config[SAYCAN][LLM_MODEL][LLAMA], device)
        elif platform == MANUAL:
            self.llm_model = LLMModelManual(self.config[SAYCAN][LLM_MODEL][MANUAL], device)
        elif platform == REPLAY:
            self.llm_model = LLMModelReplay(self.config[SAYCAN][LLM_MODEL][REPLAY], device)
        else:
            print (f'llm setup failed, cannot find llm platform {self.config[SAYCAN][PLATFORM]}')

    # call at each episode
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)
        if self.config[SAYCAN][LLM_MODEL][PLATFORM] in [MANUAL, REPLAY]:
            self.llm_model.reset()
        self.prompt_history = []
        self.response_history = []
        self.response = None
        self.prompt = None
        self.plan = LLMPlan()
        self.cur_step = 0
        self.num_prompt = 0
            
    # the external tool for sending in the prompt and get the plan
    def prompt_and_plan(self, prompt: dict, info=None):
        self.prompt = prompt
        if self.prompt[PLANNER] is not None:
            self.prompt[PLANNER][PREFIX] = self.prompt_prefix
        self._generate_plan(info=info)
        self.prompt_history.append(self.prompt)
        print ('generated plan')
        print (self.plan.print_plan())
        self.log_prompt_and_response()
        self.num_prompt += 1
        
    def get_plan(self):
        content = self.plan.get_content()
        return content
    
    def pop_next_step(self):
        next_step = self.plan.pop_next_step()
        return next_step
    
    def get_next_step_and_increase_counter(self, nl_context=None, info=None):
        step = self.plan.get_step(self.cur_step)
        if step is not None:
            self.cur_step += 1
            return step
        else:
            if self.plan_mode == ONCE:
                return None
            elif self.plan_mode in [STEP, MULTISTEP]:
                self.prompt_and_plan(nl_context, info=info)
                step = self.plan.get_step(self.cur_step)
                self.cur_step += 1
                return step
    
    def get_step(self, step_index):
        step = self.plan.get_step(step_index)
        return step
    
    # Internal plan generation functions
    # a generic planner, either plan autoregressive or once
    def _generate_plan(self, info=None):
        rooms, discovered_objects, observed_objects, observed_recs, gripped_obj = self.get_rooms_gripped_observed_objects_recs(info)
        skill_msg = self.compile_skills_msg(rooms, gripped_obj, discovered_objects, observed_objects, observed_recs)
        # then use the graph search results to prompt the planner
        prompt_planner = self._build_prompt_for_planner(self.prompt[PLANNER], skill_msg)
        self.prompt[PLANNER] = prompt_planner
        self.llm_model.generate_response(prompt_planner, info)
        self._process_llm_response_to_plan(dryrun=False)
        return self.plan.get_content()
    
    def compile_skills_msg(self, rooms, gripped_obj, discovered_objects, observed_objects, observed_recs):
        # produce a list of skills which can be executed given local observations and room exploration actions
        # go to room (~6 skills)
        # go to observed object (~2 skills)
        # look at observed object (~2 skills)
        # if holding nothing:
        #   pick up observed object (~2 skills)
        # else:
        #   place gripped object on one of the observed receptacles (~10 skills)
        # go to observed receptacle (~10 skills)
        # look at observed receptacle (~10 skills)
        # mission complete (~1 skill)
        skill_msgs = []
        room_msg = [f'go to {r}' for i, r in enumerate(rooms)]
        skill_msgs.extend(room_msg)
        if len(discovered_objects) > 0:
            go_to_obj_msg = [f'go to {o}' for i, o in enumerate(discovered_objects)]
            skill_msgs.extend(go_to_obj_msg)
            look_at_obj_msg = [f'look at {o}' for i, o in enumerate(discovered_objects)]
            skill_msgs.extend(look_at_obj_msg)
        if gripped_obj in [NOTHING, None, '']:
            # add pick up msgs
            if len(observed_objects) > 0:
                pick_up_obj_msg = [f'pick up {o}' for i, o in enumerate(observed_objects)]
                skill_msgs.extend(pick_up_obj_msg)
        else:
            # combination of the gripped object with each observed rec
            if len(observed_recs) > 0:
                # add placement msgs
                place_obj_msg = [f'place {gripped_obj} on {r}' for i, r in enumerate(observed_recs)]
                skill_msgs.extend(place_obj_msg)
        if len(observed_recs) > 0:
            go_to_rec_msg = [f'go to {r}' for i, r in enumerate(observed_recs)]
            skill_msgs.extend(go_to_rec_msg)
            look_at_rec_msg = [f'look at {r}' for i, r in enumerate(observed_recs)]
            skill_msgs.extend(look_at_rec_msg)
        skill_msgs = [m for m in skill_msgs if len(m)> 0]
        skill_msgs = [f'{m}' for m in skill_msgs]
        random.shuffle(skill_msgs)
        mission_complete_msg = ['mission complete']
        skill_msgs.extend(mission_complete_msg)
        output_skill_msg = ', '.join(skill_msgs)
        return output_skill_msg
    
    def get_rooms_gripped_observed_objects_recs(self, info):
        graph = info[GRAPH]
        gripped_obj = graph.gripped_obj
        rooms = list(graph.graph.keys())
        local_observations = graph.get_local_observations()
        # use all objects found up to now
        discovered_objects = graph.get_all_discovered_objects()
        observed_objects = local_observations[OBJ]
        # use local observations
        observed_recs = local_observations[REC]
        return rooms, discovered_objects, observed_objects, observed_recs, gripped_obj
        
        
    def _build_prompt_for_planner(self, prompt, skill_msg=''):
        # Input: prompt for planner in the form {user: x, system: y, prefix: z}
        assert (PREV_STEPS_MSG in prompt[USER])
        # add the execution history to the high-level planner
        msg = ""
        if self.planner_include_prev_steps_msg != False:
            prev_steps = self.house_logger.get_prev_steps()
            msg = "\nExecuted steps: " + ','.join(prev_steps[LOW_LEVEL][self.planner_include_prev_steps_msg])
        result = {
            USER: prompt[USER].replace(PREV_STEPS_MSG, msg).replace(SKILL_PLACEHOLDER, skill_msg),
            SYSTEM: prompt[SYSTEM],
            PREFIX: self.prompt_prefix
        }
        print ('--- Planner Prompt ---')
        for k in result:
            print (f"[{k}] {result[k]}")
        print ('----------------------')
        return result
    
    def _process_llm_response_to_plan(self, dryrun=False):
        top_nl_response = self.prompt_prefix + self.llm_model.get_top_nl_response()
        print ("--- raw llm response --- ")
        print (top_nl_response)
        print ('------------------------')
        # post-processing
        top_nl_response = top_nl_response.lower().replace(' :', ":")
        self.response = self.llm_model.get_top_nl_response().lower().replace(' :', ":")
        
        if self.cur_step > self.prompt_threshold or self.num_prompt > self.max_prompt_budget:
            top_nl_response = self.prompt_prefix + MISSION_COMPLETE
            top_nl_response = top_nl_response.lower().replace(' :', ":")
            self.response = MISSION_COMPLETE
        # processing the top response as one or multiple steps
        pattern = r"\n|step \d+\:|step\d+\:"
        # Split the string based on the pattern
        steps = [s.strip().strip('.').strip(',') for s in re.split(pattern, top_nl_response)]
        steps = [s for s in steps if len(s)>0] # exclude null results
        selected_steps = []
        if self.plan_mode == ONCE:
            selected_steps = steps[:]
        elif self.plan_mode == MULTISTEP:
            selected_steps = steps[:self.max_steps_per_prompt] # avoid cutting off the last step
        elif self.plan_mode == STEP:
            selected_steps = steps[:1]
        else:
            print (f"mode {self.plan_mode} not implemented")
        if not dryrun:
            self.add_plan_steps_to_plan(selected_steps)
        return selected_steps
                
    def add_plan_steps_to_plan(self, steps):
        self.plan.add_steps(steps)


class LLMPlan:
    # an LLMPlan is in the form of a list of steps (in str)
    def __init__(self):
        self.content = []
        
    def print_plan(self):
        result = "-------- PLAN --------\n"
        result += '\n'.join(self.content)
        result += '\n----------------------'
        return result
        
    def add_step(self, step):
        self.content.append(step)
        
    def add_steps(self, steps):
        for step in steps:
            self.content.append(step)
            
    def get_step(self, step_index):
        if step_index < len(self.content):
            return self.content[step_index]
        else:
            print (f"Step {step_index} above plan content length {len(self.content)}")
            return None
    
    def pop_next_step(self):
        if len(self.content) > 0:
            return self.content.pop(self.content, 0)
        else:
            print ('empty plan object, no step to pop')
            return None
    
    def get_content(self):
        return self.content
    
    def reset(self):
        self.content = []


# A generic llm model
class LLMModel(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.response = None
        
    @abstractmethod
    def setup(self):
        pass
        
    @abstractmethod
    def generate_response(self, prompt: dict, info=None):
        pass
    
    def get_response(self):
        return self.response
    
    def get_top_nl_response(self):
        return self.response[0]['sample']
    
# Manual model for debug
class LLMModelManual(LLMModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.mode = config.get(MODE, ORACLE)
        self.setup()
        
    def setup(self):
        print (f'manual planner model loaded')
        self.visited_rooms = {}
        
    def reset(self):
        self.visited_rooms = {}
        
    def generate_response(self, prompt: dict, info=None):
        # the generated samples should look like the following: 1. do x, 2. do y, ...
        # the info is a dictionary of the current observations
        if self.mode == ORACLE:
            correct_mapping = info[CORRECT_MAPPING]
            current_mapping = info[CURRENT_MAPPING]
            items_to_move = {k: correct_mapping[k] for k in current_mapping if current_mapping[k] not in correct_mapping[k]}
            # then we'll formulate a plan according to items to move
            plan = ''
            step_counter = 0
            for k in items_to_move:
                rec = items_to_move[k][0]
                step_counter += 1
                nav_obj = f"step {step_counter}: go to {k}"
                step_counter += 1
                look_obj = f"step {step_counter}: look at {k}"
                step_counter += 1
                pick_obj = f"step {step_counter}: pick up {k}"
                step_counter += 1
                nav_rec = f"step {step_counter}: go to {rec}"
                step_counter += 1
                look_rec = f"step {step_counter}: look at {rec}"
                step_counter += 1
                place_rec = f"step {step_counter}: place {k} on {rec}"
                plan += ' '.join([nav_obj, look_obj, pick_obj, nav_rec, look_rec, place_rec])
        elif self.mode == TEST:
            plan = ''
            step_counter = 0
            for room in info[ROOM]:
                step_counter += 1
                plan += f"step {step_counter}: go to {room}"
            for primitive_action in [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_DOWN, LOOK_UP]:
                step_counter += 1
                plan += f"step {step_counter}: {primitive_action}"
        elif self.mode == DEMO:
            house_graph = info[GRAPH].graph
            chosen_rooms = []
            for room in house_graph:
                if room not in self.visited_rooms:
                    chosen_rooms.append(room)
            if len(chosen_rooms) > 0:
                steps = []
                # start by exploring the rooms
                step_counter = 0
                for k in chosen_rooms:
                    step_counter += 1
                    nav_obj = f"step {step_counter}: go to {k}"
                    steps.append(nav_obj)
                    self.visited_rooms[k] = True
                plan = ' '.join(steps)
            else:
                # sufficient exploration, start pick and place
                correct_mapping = info[CORRECT_MAPPING]
                current_mapping = info[CURRENT_MAPPING]
                items_to_move = {k: correct_mapping[k] for k in current_mapping if current_mapping[k] not in correct_mapping[k]}
                correct_rec = None
                cur_rec = None
                for obj in items_to_move:
                    cur_rec = None
                    correct_rec = None
                    correct_recs = [rec for rec in correct_mapping[obj]][:]
                    random.shuffle(correct_recs)
                    # check if item in view and correct rec in view
                    for room in house_graph:
                        if obj in house_graph[room][OBJ]:
                            cur_rec = house_graph[room][OBJ][obj][REC]
                            break
                    for rec in correct_recs:
                        for room in house_graph:
                            if rec in house_graph[room][REC]:
                                correct_rec = rec
                                break
                        if correct_rec is not None:
                            break
                    if correct_rec is not None and cur_rec is not None:
                        break
                # then we'll formulate a plan by moving one object
                if correct_rec is None or cur_rec is None:
                    plan = "mission complete"
                else: 
                    plan = ''
                    step_counter = 1
                    nav_obj = f"step {step_counter}: go to {obj}"
                    step_counter += 1
                    look_obj = f"step {step_counter}: look at {obj}"
                    step_counter += 1
                    pick_obj = f"step {step_counter}: pick up {obj}"
                    step_counter += 1
                    nav_rec = f"step {step_counter}: go to {correct_rec}"
                    step_counter += 1
                    look_rec = f"step {step_counter}: look at {correct_rec}"
                    step_counter += 1
                    place_rec = f"step {step_counter}: place {obj} on {correct_rec}"
                    plan += ' '.join([nav_obj, look_obj, pick_obj, nav_rec, look_rec, place_rec])
        else:
            print (f'LLM Model Manual model {self.mode} not implemented, choose oracle or test')
        self.response =  [{"sample": plan, "log_probs":0}]
        return self.response


# Replay model from the logs for video generation
class LLMModelReplay(LLMModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.setup()
    
    def setup(self):
        self.log_path = self.config[LOG_PATH]
        print (f'Replay model from logs {self.log_path} loaded')
        
    def reset(self):
        self.prompt_counter = 0
        with open(self.log_path, 'r') as infile:
            text = infile.read()
            results = json.loads(text)
        self.responses_history = []
        for record in results:
            self.responses_history.append(record[LOW_LEVEL][RESPONSE])
    
    def generate_response(self, prompt: dict, info=None):
        if self.prompt_counter < len(self.responses_history):
            self.response = [{'sample': self.responses_history[self.prompt_counter], 'log_prob': 0}]
        else:
            print ('replay agent: No more logged responses, completing the episode...')
            self.response = [{'sample': 'mission complete', 'log_prob': 0}]
        self.prompt_counter += 1
        return self.response

# Openai model series
class LLMModelOpenai(LLMModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.setup()
    
    def setup(self):
        import yaml
        with open("cos_eor/configs/local/api_key.yaml", 'r') as file:
            data = yaml.safe_load(file)
        if not "key" in data:
            print ("put api key in cos_eor/configs/local/api_key.yaml as key:<key>")
        openai.api_key = data["key"]
        if 'organization' in data:
            openai.organization = data['organization']
        self.sampling_params = self.config[SAMPLING_PARAMS]
        print (f'OpenAI model {self.config[MODEL]} loaded')
        print ('Sampling params', self.sampling_params)
    
    def generate_response(self, prompt: dict, info=None):
        MAX_RETRY = 3
        RETRY_DELAY = 60
        for attempt in range(MAX_RETRY):
            try:
                if self.config[MODEL] == GPT_35_TURBO or self.config[MODEL] in ['gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613'] or GPT_4 in self.config[MODEL] or 'ft:gpt-3.5-turbo-0613:' in self.config[MODEL]:
                    self.sampling_params_without_logprobs = {k: v for k, v in self.sampling_params.items() if k != LOG_PROBS}
                    messages = [{ ROLE: SYSTEM, CONTENT: prompt[SYSTEM] }, {ROLE: USER, CONTENT: prompt[USER] + prompt[PREFIX]}]
                    print ('creating prompt resquest...')
                    print ('SAMPLING PARAMS', self.sampling_params_without_logprobs)
                    print ('MESSAGES', messages)
                    response = openai.ChatCompletion.create(model=self.config[MODEL], messages=messages, **self.sampling_params_without_logprobs)
                    generated_samples = [response['choices'][i]['message'][CONTENT] for i in range(self.sampling_params['n'])]
                    self.response =  list(sorted([{'sample': generated_samples[i], 'log_prob': 0} for i in range(len(generated_samples))], key=lambda x: x['log_prob'], reverse=True))
                elif self.config[MODEL] in [TEXT_DAVINCI_003, GPT_35_TURBO_INSTRUCT]:
                    input_prompt = prompt[SYSTEM] + prompt[USER] + prompt[PREFIX]
                    response = openai.Completion.create(engine=self.config[MODEL], prompt=input_prompt, **self.sampling_params)
                    generated_samples = [response['choices'][i]['text'] for i in range(self.sampling_params['n'])]
                    # calculate mean log prob across tokens\n",
                    mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(self.sampling_params['n'])]
                    self.response =  list(sorted([{'sample': generated_samples[i], 'log_prob': mean_log_probs[i]} for i in range(len(generated_samples))], key=lambda x: x['log_prob'], reverse=True))
                else:
                    print (f"model {self.config[MODEL]} not yet included in the llm planner (see line 650)")
                break
            except Exception as e:
                print(f"OpenAI LLM Response Generation error occurred: {e}")
                # Retry logic
                if attempt < MAX_RETRY - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Maximum retries reached, giving up.")
        return self.response
    
# Huggingface model series
class LLMModelHf(LLMModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.setup()
    
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config[MODEL])
        self.model = AutoModelForCausalLM.from_pretrained(self.config[MODEL], pad_token_id=self.tokenizer.eos_token_id, device_map="auto", max_memory=self.config[MAX_MEMORY])
        # self.model = torch.nn.DataParallel(self.model)
        # self.model.to("cuda")
        self.sampling_params = self.config[SAMPLING_PARAMS]
        self.sampling_params_without_tokens = {k: v for k, v in self.sampling_params.items() if k != MAX_TOKENS}
        print (f'Huggingface model {self.config[MODEL]} loaded')
    
    def generate_response(self, prompt: dict, info=None):
        input_prompt = prompt[SYSTEM] + prompt[USER] + prompt[PREFIX]
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[-1]
        with torch.no_grad():
            output_dict = self.model.generate(input_ids, max_length=prompt_len + self.sampling_params[MAX_TOKENS], **self.sampling_params_without_tokens)
        # discard the prompt (only take the generated text)
        generated_samples = self.tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
        # calculate per-token logprob
        vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
        token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()  # [n, length]
        # truncate each sample if it contains '\n' (the current step is finished)
        # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
        for i, sample in enumerate(generated_samples):
            stop_idx = sample.index('\n') if '\n' in sample else None
            generated_samples[i] = sample[:stop_idx]
            token_log_probs[i] = token_log_probs[i][:stop_idx]
        # calculate mean log prob across tokens
        mean_log_probs = [np.mean(token_log_probs[i]) for i in range(self.sampling_params['num_return_sequences'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        self.response = list(sorted([{'sample': generated_samples[i], 'log_prob': mean_log_probs[i]} for i in range(len(generated_samples))], key=lambda x: x['log_prob'], reverse=True))
        return self.response
        
        
# Llama model series
class LLMModelLlama(LLMModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.setup()
        
    def setup(self):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        # home_directory = os.path.expanduser("~") # for all other systems
        home_directory = os.path.join('/disk', 'scratch1', 'dhan') # for am1
        workplace_directory = os.path.join(home_directory, "workplace")
        self.config[MODEL_PATH] = os.path.join(workplace_directory, self.config[MODEL])
        print ('LLama path', self.config[MODEL_PATH])
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config[MODEL_PATH])
        self.model = LlamaForCausalLM.from_pretrained(self.config[MODEL_PATH], max_memory=self.config[MAX_MEMORY], device_map='auto', load_in_8bit=True, torch_dtype=torch.float16).bfloat16()
        self.sampling_params = self.config[SAMPLING_PARAMS]
        self.sampling_params_without_tokens = {k: v for k, v in self.sampling_params.items() if k != MAX_TOKENS}
        print (f'Llama model {self.config[MODEL]} loaded')
        
    def generate_response(self, prompt: dict, info=None):
        input_prompt = prompt[SYSTEM] + prompt[USER] + prompt[PREFIX]
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[-1]
        with torch.no_grad():
            output_dict = self.model.generate(input_ids, max_length=prompt_len + self.sampling_params[MAX_TOKENS], **self.sampling_params_without_tokens)
        # discard the prompt (only take the generated text)
        generated_samples = self.tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
        # calculate per-token logprob
        vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
        token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()  # [n, length]
        # truncate each sample if it contains '\n' (the current step is finished)
        # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
        for i, sample in enumerate(generated_samples):
            stop_idx = sample.index('\n') if '\n' in sample else None
            generated_samples[i] = sample[:stop_idx]
            token_log_probs[i] = token_log_probs[i][:stop_idx]
        # calculate mean log prob across tokens
        mean_log_probs = [np.mean(token_log_probs[i]) for i in range(self.sampling_params['num_return_sequences'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        self.response = list(sorted([{'sample': generated_samples[i], 'log_prob': mean_log_probs[i]} for i in range(len(generated_samples))], key=lambda x: x['log_prob'], reverse=True))
        return self.response