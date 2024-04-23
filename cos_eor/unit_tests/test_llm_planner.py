# unit test for the llm modules
import os
import unittest
import yaml
import sys
import argparse

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)
for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from cos_eor.utils.constants import *
from cos_eor.policy.llm_planner import LLMPlanModule, LLMModelHf


class TestLLMPlanner(unittest.TestCase):
    
    def setup(self):
        with open('cos_eor/unit_tests/test_llm_planner_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
            self.prompt = "You are a one-handed robot in a household. There are misplaced objects and receptacles. \
                Give me a concrete plan to find and pick up misplaced objects and place onto correct receptacles \
                (potentially in a different room) using your knowledge. Steps: " 
            self.planner = LLMPlanModule(self.config, 1, 0)
            
    def test_llmplanner_get_plan(self):
        self.planner.reset()
        self.planner.prompt_and_plan(self.prompt)
        print ('llm response', self.planner.llm_model.response)
        plan = self.planner.get_plan()
        print ('plan', plan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debugpy",
        default=False,
        action='store_true',
        help="Run in debug mode and wait for debugger to attach."
    )
    args = parser.parse_args()
    if args.debugpy:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))  # use the same address and port as in your VSCode config
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached!")
        
    test = TestLLMPlanner()
    test.setup()
    test.test_llmplanner_get_plan()


