# unit test for the llm modules
import os
import unittest
import yaml
import sys

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)
for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from cos_eor.utils.constants import *
from cos_eor.policy.llm_planner import LLMModelLlama, LLMModelHf, LLMModelOpenai


class TestLLMModels(unittest.TestCase):
    
    def setup(self):
        with open('cos_eor/unit_tests/test_llm_models_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
            self.prompt = "Generate a plan to make breakfast. Step 1:"
            
    def test_openai(self):
        model = LLMModelOpenai(self.config[OPENAI], 0)
        print (f"{self.config[OPENAI][MODEL]} model loaded")
        response = model.generate_response(prompt=self.prompt)
        for r in response:
            print (r)

    def test_llama(self):
        model = LLMModelLlama(self.config[LLAMA], 0)
        print (f"{self.config[LLAMA][MODEL]} model loaded")
        response = model.generate_response(prompt=self.prompt)
        for r in response:
            print (r)

    def test_hf(self):
        model = LLMModelHf(self.config[HF], 0)
        print (f"{self.config[HF][MODEL]} model loaded")
        response = model.generate_response(prompt=self.prompt)
        for r in response:
            print (r)
        

if __name__ == "__main__":
    test = TestLLMModels()
    test.setup()
    test.test_llama()


