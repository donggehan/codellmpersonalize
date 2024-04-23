from collections import defaultdict, OrderedDict
import math
import datetime
import os
import json
import magnum as mn
import numpy as np
import torch
import re
import copy

from cos_eor.utils.objects_to_byte_tensor import dec_bytes2obj
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy
from cos_eor.policy.explore import ExploreModule
from cos_eor.policy.nav import NavModule, NavTargetInfo
from cos_eor.utils.visualization import add_text, render_frame_explore_sim
from habitat.utils.visualizations.utils import images_to_video
from cos_eor.utils.constants import *

# CODES
SUC = "succeeded"
INP = "in-progress"
FAIL = "failed"
START = "start"

OBJ = "object"
REC = "receptacle"

NAV = "navigating"
PP = "pick-place"
LOOK = "look-at"
EXP = "exploring"


class HiePolicy:
    """
    Contains:
    1. The nav-agent that is to be trained for reaching goals given semantic-id
    2. The rule-based / learned exploration strategy
    3. The internal-state matrix and scoring/ranking function for ORM.

    """
    def __init__(self, envs, nav_module, rank_module, explore_module, context_module, plan_module, policy_params, task_params, debug_params=None):
        self.envs = envs
        self.nav_module = nav_module
        self.rank_module = rank_module
        self.explore_module = explore_module
        self.plan_module = plan_module
        self.policy_params = policy_params
        self.task_params = task_params
        self.context_module = context_module
        self.key_translator = KeyTranslator(simplify_keys=self.context_module.simplify_keys)
        self.house_logger = HouseLogger(plan_mode=self.plan_module.type)
        self.allowed_options = self.plan_module.allowed_options
        self.context_module.allowed_options = self.allowed_options
        if policy_params.oracle:
            self.fail_thresholds = {
                f"{NAV}-{OBJ}-{INP}": 500,  # max-steps for nav
                f"{NAV}-{REC}-{INP}": 500,  # max-steps for nav
                f"{PP}-{REC}-{INP}": 0,  # max-tries for pp
                f"{PP}-{OBJ}-{INP}": 0,  # max-tries for pp
                f"{LOOK}-{OBJ}-{INP}": 100,  # max-steps for look
                f"{LOOK}-{REC}-{INP}": 100,  # max-steps for look
                f"{EXP}-{None}-{INP}": 100,  # max-tries for exp
            }
        else:
            self.fail_thresholds = {
                f"{NAV}-{OBJ}-{INP}": 500,  # max-steps for nav
                f"{NAV}-{REC}-{INP}": 500,  # max-steps for nav
                f"{PP}-{REC}-{INP}": 0,  # max-tries for pp
                f"{PP}-{OBJ}-{INP}": 0,  # max-tries for pp
                f"{LOOK}-{OBJ}-{INP}": 20,  # max-steps for look
                f"{LOOK}-{REC}-{INP}": 20,  # max-steps for look
                f"{EXP}-{None}-{INP}": self.policy_params.explore.max_steps,  # max-tries for exp
            }
        self.reset()

        self.debug_video = False
        self.measures = {}
        self.turn_measures = {}

    def reset(self):
        # we want them to be ordered so that indices can map 1-1 to matrices
        self.rec_rooms = OrderedDict()
        self.objs = OrderedDict()
        self.bad_recs = set()
        self.room_recs = OrderedDict() # to keep track of all rooms in the household

        self.pending_rearrangements = []  # fifo queue
        self.past_rearrangements = OrderedDict()
        self.obj_key_to_sim_obj_id = {}
        self.curr_rearrangment = None  # should be (obj_idx, curr_rec_idx, curr_best_rec_idx)
        self.tracker = defaultdict(int)
        self.measures = {
            "success_look_at": 0,
            "fail_look_at": 0,
        }

        # debug
        self.obs = []
        self.raw_obs = []
        self.debug_count = 0
        self.step_count = 0

        # reset all internal components
        self.reset_state()
        self.rank_module.reset()
        self.explore_module.reset()
        self.context_module.reset(key_translator=self.key_translator, house_logger=self.house_logger)
        self.plan_module.reset(key_translator=self.key_translator, house_logger=self.house_logger)
        self.nav_module.reset()
        self.key_translator.reset()
        self.house_logger.reset()
        self.timestep = 0
        
        # added counters
        self.temp_trackers = {
            LOOK: 0,
            NAV: 0,
            PP: 0
        }

    def reset_state(self):
        self.controller_in_progress = False
        self.curr_state = {
            "act": EXP,  # one from ["nav", "pick-place", "look-at", "explore"]
            "target": None,  # one from ["obj", "rec"]
            "status": INP,  # one from ["start", "in-progress", "success", "fail"]
            "done": False # true or false
        }

    def get_current_state(self):
        return [self.curr_state]

    def _rearrange_sort_key(self, rearrangement, scores):
        oi, cmi, (ssi, ss) = rearrangement
        order = self.policy_params.rearrange_order
        if order == "discovery":
            return oi
        elif order == "score_gain":
            cur_score = scores[cmi, oi]
            return cur_score - ss[0]
        elif order == "agent_distance":
            return self._get_agent_obj_dist(ssi[0], REC, "l2")
        elif order == "obj_distance":
            return self._get_obj_obj_dist(cmi, REC, ssi[0], REC, "l2")
        else:
            raise ValueError

    def skip_initial_bad_recs(self, rearrangement):
        oi, cmi, (ssi, ss) = rearrangement
        start_idx = 0
        for ri in ssi:
            past_rearrangement_status = self.past_rearrangements.get((oi, cmi, ri))
            if past_rearrangement_status is not None:
                past_rearrangement_status = past_rearrangement_status[1]
            if past_rearrangement_status != FAIL and ri not in self.bad_recs:
                break
            start_idx += 1
        return start_idx

    def get_rearrangements(self):
        # Comment: this contains all info about the terms and concepts
        if len(self.objs) == 0 or len(self.rec_rooms) == 0:
            # Comment: no object found and no recs and rooms found
            return
        # sorted receptacle indices from high-low similarity scores
        combined_scores = self.rank_module.scores + self.rank_module.room_scores
        combined_scores[self.rank_module.scores < self.policy_params.score_threshold] = 0
        sort_scores_inds = (combined_scores * -1).argsort(axis=0).T
        sort_scores = np.array([self.rank_module.scores[ssi, idx] for idx, ssi in enumerate(sort_scores_inds)])

        # Comment: this is the objects' current matched recs
        curr_match_keys = [obj["rec"] for obj in self.objs.values()]
        # Comment: a list of seen receptacles up to now
        curr_recs_keys = self.get_seen("obj_key", types=["rec"])
        # Comment: a map between seen (curr) recs keys and their ids
        curr_recs_key_to_idx = {key: idx for idx, key in enumerate(curr_recs_keys)}
        # Comment: seen recs indices
        curr_match_inds = [curr_recs_key_to_idx[rk] if rk in curr_recs_keys else -1 for rk in curr_match_keys]

        rearrangements = []
        # match and return
        for obj_idx, (cmi, ssi, ss) in enumerate(zip(
                curr_match_inds,
                sort_scores_inds,
                sort_scores,
        )):
            if cmi == -1:
                # Comment: rec is not seen yet
                continue
            # skip if we are already committed
            if self.curr_rearrangment is not None and obj_idx == self.curr_rearrangment[0]:
                continue
            start_idx = self.skip_initial_bad_recs((obj_idx, cmi, (ssi, ss)))
            ssi, ss = ssi[start_idx:], ss[start_idx:]

            curr_score = self.rank_module.scores[cmi, obj_idx]
            if (
                ss[0] > self.policy_params.score_threshold and
                curr_score <= self.policy_params.score_threshold and
                ssi.size > 0 and
                cmi != ssi[0]
            ):
                rearrangements.append((obj_idx, cmi, (ssi, ss)))

        # sort rearrangements by scheme in config
        rearrangements.sort(key=lambda r: self._rearrange_sort_key(r, combined_scores))

        # merge with current-list
        pending_rearrangements_map = {pr[0]: idx for idx, pr in enumerate(self.pending_rearrangements)}
        for nr in rearrangements:
            oi, _, _ = nr
            # remove stale matches if better ones exist
            if oi in pending_rearrangements_map:
                pr_idx = pending_rearrangements_map[oi]
                self.pending_rearrangements[pr_idx] = nr
            else:
                self.pending_rearrangements.append(nr)

    def print_pending(self, log=True):
        self.debug_count += 1
        for idx, pr in enumerate(self.pending_rearrangements):
            oi, cmi, cbi, cbs = pr
            ok = self.get_value(oi, OBJ, "obj_key")
            crk = self.get_value(cmi, REC, "obj_key")
            brk = self.get_value(cbi, REC, "obj_key")
            if self.debug_count % 10 == 0 or log:
                self.log(f"R-{idx}: move obj {ok} from {crk} to {brk}")

    def assert_consistency(self):
        assert self.curr_state["status"] in [INP, FAIL, SUC, None]
        assert self.curr_state["target"] in [OBJ, REC, PRIMITIVE, None]
        if self.policy_params.explore.type == "oracle":
            assert self.curr_state["act"] in [LOOK, PP, NAV, EXP, None]
        else:
            assert self.curr_state["act"] in [LOOK, PP, NAV, EXP]
        # ensure every obj-id in pending rearrangments in unique
        # ensure curr_state and curr_rearrangment consistency

    def track(self, only_return=False, global_state=False):
        status_key = f"{self.curr_state['act']}-{self.curr_state['target']}-{self.curr_state['status']}"
        if global_state:
            track_key = status_key
        else:
            track_key = f"{status_key}-{str(self.curr_rearrangment)}"
        if not only_return:
            self.tracker[track_key] += 1
        else:
            return self.tracker[track_key]

    def assert_threshold(self, global_state=False):
        status_key = f"{self.curr_state['act']}-{self.curr_state['target']}-{self.curr_state['status']}"
        return self.fail_thresholds[status_key] >= self.track(only_return=True, global_state=global_state)

    def reset_tracker_current_state(self):
        tracker_key = f"{self.curr_state['act']}-{self.curr_state['target']}-{self.curr_state['status']}"
        self.tracker[tracker_key] = 0
        
        
    def load_next_state_llm_planner(self):
        # if the agent haven't completed the last step, continue
        if self.controller_in_progress:
            self.curr_state["done"] = False
            return
        nl_context = self.context_module.get_nl_context()
        # else (agent completed nav + pick/place), assign the next step from the planner
        if len(self.plan_module.prompt_history) == 0:
            # no plan generated yet, generate plan once
            if self.plan_module.config[SINGLE][LLM_MODEL][PLATFORM] == MANUAL:
                info = self.context_module.get_debug_info()
                self.plan_module.prompt_and_plan(nl_context, info=info)
            else:
                self.plan_module.prompt_and_plan(nl_context)
        explore_type = self.policy_params.explore.type
        if explore_type == "oracle":
            self.log("llm assigning new task!")
            
            fail_counter = 0 # reset to 0 as soon as a valid step gets executed, else +1
            while True:
                nl_context = self.context_module.get_nl_context()
                if self.plan_module.config[SINGLE][LLM_MODEL][PLATFORM] == MANUAL:
                    info = self.context_module.get_debug_info()
                    next_step = self.plan_module.get_next_step_and_increase_counter(nl_context, info=info)
                else:
                    next_step = self.plan_module.get_next_step_and_increase_counter(nl_context)
                self.log(f"--- Next Step: {next_step} ---")
            
                if next_step is None or next_step == MISSION_COMPLETE or fail_counter > self.plan_module.fail_threshold:
                    if next_step == MISSION_COMPLETE:
                        self.open_execution_record_and_snapshot_graph(next_step.lower(), {})
                        self.close_execution_record_and_summarise_graph(SKIP)
                    print ("plan finished, next step", next_step, 'failed attempts', fail_counter)
                    self.curr_state["act"] = None
                    self.curr_state["done"] = True
                    self.house_logger.summarise_record()
                    break

                plan_action, target_obj, target_rec, target_type, flag = self.plan_module.postprocess(next_step, self.context_module.get_observed())
                self.open_execution_record_and_snapshot_graph(next_step.lower(), {ACTION: plan_action, TARGET_OBJ: target_obj, TARGET_REC: target_rec, TARGET_TYPE: target_type})
                if flag == SUC:
                    try:
                        if target_type != PRIMITIVE:
                            # if primitive action, target_obj represents the action index
                            if target_obj is not None:
                                target_obj = self.get_oi_from_obj_key(target_obj, type=OBJ)
                            if target_rec is not None:
                                target_rec = self.get_oi_from_obj_key(target_rec, type=REC)

                    except KeyError as e:
                        # failure case example: None-door_x.urdf rec 
                        print (f'key error {e}')
                        fail_counter += 1
                        self.curr_state["done"] = False
                        self.close_execution_record_and_summarise_graph(SKIP)
                        continue

                    self.curr_state["act"] = plan_action  # NAV, LOOK or PP
                    self.curr_state["target"] = target_type  # OBJ or REC
                    self.curr_state["status"] = INP
                    self.curr_state['target_obj'] = target_obj  # may be left None, if not applicable
                    self.curr_state['target_rec'] = target_rec  # may be left None, if not applicable
                    self.curr_state["done"] = False
                    break  # Exit the loop when a successful step is processed

                elif flag == FAIL:
                    # If the flag is FAIL, the loop will continue to get the next step
                    fail_counter += 1
                    self.curr_state["done"] = False
                    self.close_execution_record_and_summarise_graph(SKIP)
                    continue
        else:
            print (f"explore type {explore_type} with LLM zeroshot agent not implemented")
            
    # Helper function for communicating with house_logger
    def open_execution_record_and_snapshot_graph(self, step_raw, step_parsed):
        self.context_module.snapshot_house_graph()
        self.house_logger.open_execution_record(step_raw, step_parsed, self.context_module.current_mapping, self.context_module.correct_mapping)

    # Helper function for communicating with house_logger
    def close_execution_record_and_summarise_graph(self, flag):
        if flag == SKIP:
            self.house_logger.close_execution_record(SKIP, [], [], {}, self.context_module.current_mapping)
        elif flag == SUC:
            discovered_obj, discovered_recs, moved_obj = self.context_module.diff_house_graph()
            self.house_logger.close_execution_record(SUC, discovered_obj, discovered_recs, moved_obj, self.context_module.current_mapping)
        elif flag == FAIL:
            discovered_obj, discovered_recs, moved_obj = self.context_module.diff_house_graph()
            self.house_logger.close_execution_record(FAIL, discovered_obj, discovered_recs, moved_obj, self.context_module.current_mapping)

    def load_next_state_llm_planner_adapter(self):
        # if the agent haven't completed the last step, continue
        if self.controller_in_progress:
            self.curr_state["done"] = False
            return
        
        # else (agent completed nav + pick/place), assign the next step from the planner
        nl_context = self.context_module.get_nl_context()
        if len(self.plan_module.prompt_history) == 0:
            # no plan generated yet, generate plan once
            if self.plan_module.config[HIGH_LEVEL][LLM_MODEL][PLATFORM] == MANUAL:
                info = self.context_module.get_debug_info()
                self.plan_module.prompt_and_plan(nl_context, info=info)
            else:
                self.plan_module.prompt_and_plan(nl_context)
        explore_type = self.policy_params.explore.type
        if explore_type == "oracle":
            self.log("llm assigning new task!")
            # if next step already planned, get it
            # if not, prompt and plan, if MISSION_COMPLETE, return None
            fail_counter = 0 # reset to 0 as soon as a valid step gets executed, else +1
            while True:
                nl_context = self.context_module.get_nl_context()
                next_step = self.plan_module.get_next_step_and_increase_counter(nl_context)
                self.log(f"--- Next Step: {next_step} ---")
            
                if next_step is None or next_step == MISSION_COMPLETE or fail_counter > self.plan_module.fail_threshold:
                    if next_step == MISSION_COMPLETE:
                        self.open_execution_record_and_snapshot_graph(next_step.lower(), {})
                        self.close_execution_record_and_summarise_graph(SKIP)
                    print ("plan finished, next step", next_step, 'failed attempts', fail_counter)
                    self.curr_state["act"] = None
                    self.curr_state["done"] = True
                    self.house_logger.summarise_record()
                    break

                plan_action, target_obj, target_rec, target_type, flag = self.plan_module.postprocess(next_step, self.context_module.get_observed())
                self.open_execution_record_and_snapshot_graph(next_step.lower(), {ACTION: plan_action, TARGET_OBJ: target_obj, TARGET_REC: target_rec, TARGET_TYPE: target_type})
                if flag == SUC:
                    try:
                        if target_type != PRIMITIVE:
                            # if primitive action, target_obj represents the action index
                            if target_obj is not None:
                                target_obj = self.get_oi_from_obj_key(target_obj, type=OBJ)
                            if target_rec is not None:
                                target_rec = self.get_oi_from_obj_key(target_rec, type=REC)
                    
                    except KeyError as e:
                        # failure case example: None-door_x.urdf rec 
                        print (f'key error {e}')
                        fail_counter += 1
                        self.curr_state["done"] = False
                        self.close_execution_record_and_summarise_graph(SKIP)
                        continue

                    self.curr_state["act"] = plan_action  # NAV, LOOK or PP
                    self.curr_state["target"] = target_type  # OBJ or REC
                    self.curr_state["status"] = INP
                    self.curr_state['target_obj'] = target_obj  # may be left None, if not applicable
                    self.curr_state['target_rec'] = target_rec  # may be left None, if not applicable
                    self.curr_state["done"] = False
                    break  # Exit the loop when a successful step is processed

                elif flag == FAIL:
                    # If the flag is FAIL, the loop will continue to get the next step
                    fail_counter += 1
                    self.curr_state["done"] = False
                    self.close_execution_record_and_summarise_graph(SKIP)
                    continue
        else:
            print (f"explore type {explore_type} with LLM zeroshot agent not implemented")
            
    def load_next_state_llm_planner_sayplan(self):
        # if the agent haven't completed the last step, continue
        if self.controller_in_progress:
            self.curr_state["done"] = False
            return
        nl_context = self.context_module.get_nl_context()
        # else (agent completed nav + pick/place), assign the next step from the planner
        if len(self.plan_module.prompt_history) == 0:
            # no plan generated yet, generate plan once
            info = self.context_module.get_debug_info()
            self.plan_module.prompt_and_plan(nl_context, info=info)
        explore_type = self.policy_params.explore.type
        if explore_type == "oracle":
            self.log("llm assigning new task!")
            
            fail_counter = 0 # reset to 0 as soon as a valid step gets executed, else +1
            while True:
                nl_context = self.context_module.get_nl_context()
                info = self.context_module.get_debug_info()
                next_step = self.plan_module.get_next_step_and_increase_counter(nl_context, info=info)
                self.log(f"--- Next Step: {next_step} ---")
            
                if next_step is None or next_step == MISSION_COMPLETE or fail_counter > self.plan_module.fail_threshold:
                    if next_step == MISSION_COMPLETE:
                        self.open_execution_record_and_snapshot_graph(next_step.lower(), {})
                        self.close_execution_record_and_summarise_graph(SKIP)
                    print ("plan finished, next step", next_step, 'failed attempts', fail_counter)
                    self.curr_state["act"] = None
                    self.curr_state["done"] = True
                    self.house_logger.summarise_record()
                    break

                plan_action, target_obj, target_rec, target_type, flag = self.plan_module.postprocess(next_step, self.context_module.get_observed())
                self.open_execution_record_and_snapshot_graph(next_step.lower(), {ACTION: plan_action, TARGET_OBJ: target_obj, TARGET_REC: target_rec, TARGET_TYPE: target_type})
                if flag == SUC:
                    try:
                        if target_type != PRIMITIVE:
                            # if primitive action, target_obj represents the action index
                            if target_obj is not None:
                                target_obj = self.get_oi_from_obj_key(target_obj, type=OBJ)
                            if target_rec is not None:
                                target_rec = self.get_oi_from_obj_key(target_rec, type=REC)

                    except KeyError as e:
                        # failure case example: None-door_x.urdf rec 
                        print (f'key error {e}')
                        fail_counter += 1
                        self.curr_state["done"] = False
                        self.close_execution_record_and_summarise_graph(SKIP)
                        continue

                    self.curr_state["act"] = plan_action  # NAV, LOOK or PP
                    self.curr_state["target"] = target_type  # OBJ or REC
                    self.curr_state["status"] = INP
                    self.curr_state['target_obj'] = target_obj  # may be left None, if not applicable
                    self.curr_state['target_rec'] = target_rec  # may be left None, if not applicable
                    self.curr_state["done"] = False
                    break  # Exit the loop when a successful step is processed

                elif flag == FAIL:
                    # If the flag is FAIL, the loop will continue to get the next step
                    fail_counter += 1
                    self.curr_state["done"] = False
                    self.close_execution_record_and_summarise_graph(SKIP)
                    continue
        else:
            print (f"explore type {explore_type} with LLM zeroshot agent not implemented")

    def load_next_state_llm_planner_saycan(self):
        # if the agent haven't completed the last step, continue
        if self.controller_in_progress:
            self.curr_state["done"] = False
            return
        nl_context = self.context_module.get_nl_context()
        # else (agent completed nav + pick/place), assign the next step from the planner
        if len(self.plan_module.prompt_history) == 0:
            # no plan generated yet, generate plan once
            info = self.context_module.get_debug_info()
            self.plan_module.prompt_and_plan(nl_context, info=info)
        explore_type = self.policy_params.explore.type
        if explore_type == "oracle":
            self.log("llm assigning new task!")
            
            fail_counter = 0 # reset to 0 as soon as a valid step gets executed, else +1
            while True:
                nl_context = self.context_module.get_nl_context()
                info = self.context_module.get_debug_info()
                next_step = self.plan_module.get_next_step_and_increase_counter(nl_context, info=info)
                self.log(f"--- Next Step: {next_step} ---")
            
                if next_step is None or next_step == MISSION_COMPLETE or fail_counter > self.plan_module.fail_threshold:
                    if next_step == MISSION_COMPLETE:
                        self.open_execution_record_and_snapshot_graph(next_step.lower(), {})
                        self.close_execution_record_and_summarise_graph(SKIP)
                    print ("plan finished, next step", next_step, 'failed attempts', fail_counter)
                    self.curr_state["act"] = None
                    self.curr_state["done"] = True
                    self.house_logger.summarise_record()
                    break

                plan_action, target_obj, target_rec, target_type, flag = self.plan_module.postprocess(next_step, self.context_module.get_observed())
                self.open_execution_record_and_snapshot_graph(next_step.lower(), {ACTION: plan_action, TARGET_OBJ: target_obj, TARGET_REC: target_rec, TARGET_TYPE: target_type})
                if flag == SUC:
                    try:
                        if target_type != PRIMITIVE:
                            # if primitive action, target_obj represents the action index
                            if target_obj is not None:
                                target_obj = self.get_oi_from_obj_key(target_obj, type=OBJ)
                            if target_rec is not None:
                                target_rec = self.get_oi_from_obj_key(target_rec, type=REC)

                    except KeyError as e:
                        # failure case example: None-door_x.urdf rec 
                        print (f'key error {e}')
                        fail_counter += 1
                        self.curr_state["done"] = False
                        self.close_execution_record_and_summarise_graph(SKIP)
                        continue

                    self.curr_state["act"] = plan_action  # NAV, LOOK or PP
                    self.curr_state["target"] = target_type  # OBJ or REC
                    self.curr_state["status"] = INP
                    self.curr_state['target_obj'] = target_obj  # may be left None, if not applicable
                    self.curr_state['target_rec'] = target_rec  # may be left None, if not applicable
                    self.curr_state["done"] = False
                    break  # Exit the loop when a successful step is processed

                elif flag == FAIL:
                    # If the flag is FAIL, the loop will continue to get the next step
                    fail_counter += 1
                    self.curr_state["done"] = False
                    self.close_execution_record_and_summarise_graph(SKIP)
                    continue
        else:
            print (f"explore type {explore_type} with LLM zeroshot agent not implemented")
    
    # Helper function for communicating with house_logger
    def open_execution_record_and_snapshot_graph(self, step_raw, step_parsed):
        self.context_module.snapshot_house_graph()
        self.house_logger.open_execution_record(step_raw, step_parsed, self.context_module.current_mapping, self.context_module.correct_mapping)

    # Helper function for communicating with house_logger
    def close_execution_record_and_summarise_graph(self, flag):
        if flag == SKIP:
            self.house_logger.close_execution_record(SKIP, [], [], {}, self.context_module.current_mapping)
        elif flag == SUC:
            discovered_obj, discovered_recs, moved_obj = self.context_module.diff_house_graph()
            self.house_logger.close_execution_record(SUC, discovered_obj, discovered_recs, moved_obj, self.context_module.current_mapping)
        elif flag == FAIL:
            discovered_obj, discovered_recs, moved_obj = self.context_module.diff_house_graph()
            self.house_logger.close_execution_record(FAIL, discovered_obj, discovered_recs, moved_obj, self.context_module.current_mapping)

    def load_next_state(self):
        if self.curr_rearrangment is not None:
            return

        explore_type = self.policy_params.explore.type
        if explore_type == "phasic":
            if self.curr_state["act"] == EXP and self.curr_state["status"] == INP:
                return
            elif len(self.pending_rearrangements) > 0:
                self.log("rearranging now!")
                self.curr_rearrangment = self.pending_rearrangements.pop(0)
                self.curr_state["act"] = NAV
                self.curr_state["target"] = OBJ
                self.curr_state["status"] = INP
            elif len(self.pending_rearrangements) == 0:
                if self.curr_state["act"] != EXP:
                    self.log("exploring now!")
                    self.curr_state["act"] = EXP
                    self.curr_state["target"] = None
                    self.curr_state["status"] = INP
                    self.explore_module.reset_steps_since_new_area()
                    self.reset_tracker_current_state()
        elif explore_type == "oracle":
            if len(self.pending_rearrangements) > 0:
                self.log("rearranging now!")
                self.curr_rearrangment = self.pending_rearrangements.pop(0)
                self.curr_state["act"] = NAV
                self.curr_state["target"] = OBJ
                self.curr_state["status"] = INP
            else:
                self.curr_state["act"] = None
        else:
            raise ValueError

    def dump_vid(self, video_path, video_name, raw_numpy=False):
        frames = []
        for raw_obs in self.raw_obs:
            observations, text_logs = raw_obs["obs"], raw_obs["text"]
            frame = self.create_frame(observations)
            self.add_text_logs_to_frame(frame, text_logs)
            frames.append(frame)
        video_name += f"-frames_{len(frames)}"
        if raw_numpy:
            np.save(video_path/video_name, frames)
            self.log(f"Dumped: {video_name}.npy frames")
        else:
            images_to_video(frames, video_path, video_name, fps=4)
            self.log(f"Dumped: {video_name}.mp4 video")


    def wrap_action(self, action):
        self.step_count += 1
        if self.step_count % 100 == 0:
            print(f"Num steps: {self.step_count}")
        return [{"action": action}]

    def loop_action(self, observations):
        action = None
        while action is None:
            self.load_next_state()
            if self.curr_state["act"] is None:
                action = {"action": 0}
            elif self.curr_state["act"] == NAV:
                action = self.nav(observations)
            elif self.curr_state["act"] == LOOK:
                action = self.look_at(observations)
            elif self.curr_state["act"] == PP:
                action = self.pick_place(observations)
            else:
                assert self.curr_state["act"] == EXP
                action = self.explore(observations)
        else:
            return self.wrap_action(action)
    
    def loop_action_llm(self, observations, prev_action):
        action = prev_action
        # Here handle None case when flag success
        # when flag success, we should feedback the info to planner, and get the next step
        while action['action'] is None:
            if FLAG in action:
                if action[FLAG] == SUC:
                    self.close_execution_record_and_summarise_graph(SUC)
                if action[FLAG] == FAIL:
                    self.close_execution_record_and_summarise_graph(FAIL)
            if self.plan_module.name == LLMZEROSHOT:
                self.load_next_state_llm_planner()
            elif self.plan_module.name == LLMADAPTER:
                self.load_next_state_llm_planner_adapter()
            elif self.plan_module.name == SAYPLAN:
                self.load_next_state_llm_planner_sayplan()
            elif self.plan_module.name == SAYCAN:
                self.load_next_state_llm_planner_saycan()
            else:
                print (f"Plan module {self.plan_module.name} not implemented by llm loop action")
            if self.curr_state["act"] is None:
                action = {"action": 0, "flag": DONE}
            elif self.curr_state["act"] == NAV:
                action = self.atomic_nav(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
            elif self.curr_state["act"] == LOOK:
                action = self.atomic_look_at(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
            elif self.curr_state["act"] == PP:
                action = self.atomic_pick_place(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
            else:
                print (f"action looping case self.curr_state not handled {self.curr_state['act']}")
        else:
            return action
        
    def write_house_logs_to_file(self, output_file_path):
        self.house_logger.write_log_to_file(output_file_path)
    
    def create_text_logs(self, action, cos_eor_sensor):
        if "action" not in action[0] or "action" not in action[0]["action"]:
            import pdb
            pdb.set_trace()
        action = action[0]["action"]["action"]
        possible_actions = self.task_params.POSSIBLE_ACTIONS
        action_text = possible_actions[action]

        gripped_object_id = cos_eor_sensor["gripped_object_id"]
        gripped_object_key = cos_eor_sensor["sim_obj_id_to_obj_key"].get(gripped_object_id)

        obj_key = None
        pos = (0, 0, 0)
        if self.curr_rearrangment is not None:
            oi, cmi, (ssi, ss) = self.curr_rearrangment
            if self.curr_state["target"] == OBJ:
                obj_key = self.get_value(oi, OBJ, "obj_key")
                ep_idx = cos_eor_sensor["objs_keys"].index(obj_key)
                pos = cos_eor_sensor["objs_pos"][ep_idx]
            else:
                obj_key = self.get_value(ssi[0], REC, "obj_key")
                ep_idx = cos_eor_sensor["recs_keys"].index(obj_key)
                pos = cos_eor_sensor["recs_pos"][ep_idx]

        lines = [
            f"Gripped Key: {gripped_object_key}",
            f"Target Pos: {pos[0]:.3f}, {pos[2]:.3f}",
            f"Target Key: {obj_key}",
            f"Action: {action_text}",
            f"State: {self.curr_state['act']}",
        ]

        return lines

    def add_text_logs_to_frame(self, frame, lines):
        cur_line_y = 5
        line_space = 20
        for line in reversed(lines):
            add_text(frame, line, (0, frame.shape[0] - cur_line_y))
            cur_line_y += line_space

    def create_frame(self, observations):
        frame = render_frame_explore_sim(observations)
        return frame

    def cache_raw_obs(self, observations, action):
        text_logs = self.create_text_logs(action, observations["cos_eor"][0])
        self.raw_obs.append({"obs": observations, "text": text_logs})
        
    def increment_timestep(self):
        self.timestep += 1
        self.house_logger.timestep = self.timestep

    def act(
        self,
        observations,
    ):
        self.turn_measures = {
            "seen_area": observations["seen_area"][0][0].item()
        }

        # decode task-sensor info
        observations["cos_eor"] = [dec_bytes2obj(obs) for obs in observations["cos_eor"]]
        self.update(observations)
        self.rank_module.rerank(observations["cos_eor"][0], self.rec_rooms, self.objs, True)
        self.explore_module.update(observations)
        
        
        if self.plan_module.name == ORACLE:
            # build new rearrangements
            self.get_rearrangements()
            self.assert_consistency()

            # next-state
            self.load_next_state()

            # ongoing rearrangement
            action = None
            if self.curr_rearrangment is not None:
                if self.curr_state["act"] == NAV:
                    action = self.nav(observations)
                    if action is not None:
                        action = self.wrap_action(action)
                    else:
                        action = self.loop_action(observations)
                elif self.curr_state["act"] == LOOK:
                    action = self.look_at(observations)
                    if action is not None:
                        action = self.wrap_action(action)
                    else:
                        action = self.loop_action(observations)
                elif self.curr_state["act"] == PP:
                    action = self.pick_place(observations)
                    if action is not None:
                        action = self.wrap_action(action)
                    else:
                        action = self.loop_action(observations)
                else:
                    raise ValueError
            else:
                # explore if there are no pending rearrangements
                if self.curr_state["act"] == EXP:
                    action = self.explore(observations)
                    if action is not None:
                        action = self.wrap_action(action)
                    else:
                        action = self.loop_action(observations)
                else:
                    assert (
                        self.policy_params.explore.type == "oracle" and
                        self.curr_state["act"] is None
                    )
                    action = self.wrap_action({"action": 0})
            self.cache_raw_obs(observations, action)
            self.increment_timestep()
            return action, [self.turn_measures]

        elif self.plan_module.name == LLMZEROSHOT:
            # Assume: the whole world of observations are given upfront and no exploration needed.
            # Hence currently the llm zeroshot model only works with oracle explorer
            assert (
                self.policy_params.explore.type == "oracle"
            )

            self.load_next_state_llm_planner()

            # ongoing rearrangement
            action = None
            result = None
            
            if not self.curr_state["done"]:
                if self.curr_state["act"] == NAV:
                    result = self.atomic_nav(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == LOOK:
                    result = self.atomic_look_at(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == PP:
                    result = self.atomic_pick_place(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                else:
                    print (f"self.curr_state not in the right form: {self.curr_state['act']}")
                    raise ValueError
                
            if result is None:
                # this may happen when the first step generated by the planner is mission_complete
                result = {"action": 0, "flag": DONE}
            
            if result['flag'] == INP:
                self.controller_in_progress = True # low-level controller working
            elif result['flag'] in [SUC, FAIL]:
                self.controller_in_progress = False # low-level controller idle
            elif result['flag'] == DONE:
                # plan complete
                pass
            action = result
            wrapped_action = self.wrap_action(action)
            self.cache_raw_obs(observations, wrapped_action)
            self.increment_timestep()
            return wrapped_action, [self.turn_measures]
        
        elif self.plan_module.name == LLMADAPTER:
            # Assume: the whole world of observations are given upfront and no exploration needed.
            assert (
                self.policy_params.explore.type == "oracle"
            )

            self.load_next_state_llm_planner_adapter()

            # ongoing rearrangement
            action = None
            result = None
            
            if not self.curr_state["done"]:
                if self.curr_state["act"] == NAV:
                    result = self.atomic_nav(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == LOOK:
                    result = self.atomic_look_at(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == PP:
                    result = self.atomic_pick_place(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                else:
                    print (f"self.curr_state not in the right form: {self.curr_state['act']}")
                    raise ValueError
                
            if result is None:
                # this may happen when the first step generated by the planner is mission_complete
                result = {"action": 0, "flag": DONE}
            
            if result['flag'] == INP:
                self.controller_in_progress = True # low-level controller working
            elif result['flag'] in [SUC, FAIL]:
                self.controller_in_progress = False # low-level controller idle
            elif result['flag'] == DONE:
                # plan complete
                pass
            action = result
            wrapped_action = self.wrap_action(action)
            self.cache_raw_obs(observations, wrapped_action)
            self.increment_timestep()
            return wrapped_action, [self.turn_measures]
        
        elif self.plan_module.name == SAYPLAN:
            # Assume: the whole world of observations are given upfront and no exploration needed.
            # Hence currently the llm zeroshot model only works with oracle explorer
            assert (
                self.policy_params.explore.type == "oracle"
            )

            self.load_next_state_llm_planner_sayplan()

            # ongoing rearrangement
            action = None
            result = None
            
            if not self.curr_state["done"]:
                if self.curr_state["act"] == NAV:
                    result = self.atomic_nav(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == LOOK:
                    result = self.atomic_look_at(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == PP:
                    result = self.atomic_pick_place(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                else:
                    print (f"self.curr_state not in the right form: {self.curr_state['act']}")
                    raise ValueError
                
            if result is None:
                # this may happen when the first step generated by the planner is mission_complete
                result = {"action": 0, "flag": DONE}
            
            if result['flag'] == INP:
                self.controller_in_progress = True # low-level controller working
            elif result['flag'] in [SUC, FAIL]:
                self.controller_in_progress = False # low-level controller idle
            elif result['flag'] == DONE:
                # plan complete
                pass
            action = result
            wrapped_action = self.wrap_action(action)
            self.cache_raw_obs(observations, wrapped_action)
            self.increment_timestep()
            return wrapped_action, [self.turn_measures]
        
        elif self.plan_module.name == SAYCAN:
            # Assume: the whole world of observations are given upfront and no exploration needed.
            # Hence currently the llm zeroshot model only works with oracle explorer
            assert (
                self.policy_params.explore.type == "oracle"
            )

            self.load_next_state_llm_planner_saycan()

            # ongoing rearrangement
            action = None
            result = None
            
            if not self.curr_state["done"]:
                if self.curr_state["act"] == NAV:
                    result = self.atomic_nav(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == LOOK:
                    result = self.atomic_look_at(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                elif self.curr_state["act"] == PP:
                    result = self.atomic_pick_place(observations, self.curr_state["target_obj"], self.curr_state["target_rec"],self.curr_state["target"])
                    if result["action"] is not None:
                        action = result
                    else:
                        # action failed or succeeded
                        self.controller_in_progress = False
                        result = self.loop_action_llm(observations, result)
                else:
                    print (f"self.curr_state not in the right form: {self.curr_state['act']}")
                    raise ValueError
                
            if result is None:
                # this may happen when the first step generated by the planner is mission_complete
                result = {"action": 0, "flag": DONE}
            
            if result['flag'] == INP:
                self.controller_in_progress = True # low-level controller working
            elif result['flag'] in [SUC, FAIL]:
                self.controller_in_progress = False # low-level controller idle
            elif result['flag'] == DONE:
                # plan complete
                pass
            action = result
            wrapped_action = self.wrap_action(action)
            self.cache_raw_obs(observations, wrapped_action)
            self.increment_timestep()
            return wrapped_action, [self.turn_measures]
        
        else:
            print (f"Error: no plan module named {self.plan_module.name} implemented.")

    def explore(self, observations):
        did_steps_exceed_threshold = (
            not self.assert_threshold(global_state=True) or
            self.explore_module.steps_since_new_area >= self.policy_params.explore.max_steps_since_new_area
        )
        if did_steps_exceed_threshold and len(self.pending_rearrangements) > 0:
            self.curr_state["status"] = SUC
            return None
        action = self.explore_module.act(observations)
        self.track(global_state=True)
        return action

    def _signed_angle(self, a: mn.Vector2, b: mn.Vector2) -> float:
        return math.atan2(a.x*b.y - a.y*b.x, mn.math.dot(a, b))

    def _look_at_pos(self, cos_eor_sensor, obj_pos):
        center_ray = cos_eor_sensor["camera_center_ray"]
        center_ray_origin = mn.Vector3(center_ray["origin"])
        center_ray_dir = mn.Vector3(center_ray["direction"])

        obj_pos = (mn.Vector3(obj_pos) - center_ray_origin).normalized()

        # calculate difference between current camera gaze direction
        # and gaze direction needed to look at object
        y_rot = self._signed_angle(center_ray_dir.xz, obj_pos.xz)

        # difference in angles of elevation between agent and object
        up_vec = mn.Vector3(0, 1, 0)
        up_vec_yz = up_vec.yz.normalized()
        elevation_diff = float(
            mn.math.angle(center_ray_dir.yz.normalized(), up_vec_yz) - \
            mn.math.angle(obj_pos.yz.normalized(), up_vec_yz)
        )

        # return action that reduces the larger of the angle differences
        if abs(y_rot) > abs(elevation_diff):
            if y_rot < 0:
                return 2 # left
            else:
                return 3 # right
        else:
            if elevation_diff > 0:
                return 4 # up
            else:
                return 5 # down

    def look_at(self, observations):
        # comment: objects, their current recs, and its actual matching rec and a matching score
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        cbi = ssi[0]  # current-best index

        cos_eor_sensor = observations["cos_eor"][0]
        # oi is the object index
        obj_key = self.get_value(oi, OBJ, "obj_key")

        # the agent will look for a rec, 
        # if looking for object -> find where it is on, if looking for a rec, find it
        if self.curr_state["target"] == OBJ:
            # the agent is looking for an object
            # cmi is the rec index where the object is currently on, iid is the rec id
            rec_key = self.get_value(cmi, REC, "obj_key")
            iid = self.get_value(cmi, REC, "iid")
        elif self.curr_state["target"] == REC:
            # the agent is going for a matching receptacle (after picking up the object)
            # cbi would be the best match for this picked object
            rec_key = self.get_value(cbi, REC, "obj_key")
            iid = self.get_value(cbi, REC, "iid")
        else:
            raise ValueError
        # index of the rec, found by key
        ep_idx = cos_eor_sensor["recs_keys"].index(rec_key)
        # find the position of the rec
        pos = cos_eor_sensor["recs_pos"][ep_idx]

        # check if obj/rec is within sensor frame
        visible_ids = torch.unique(observations["semantic"][0]).tolist()
        # if visible, then look at is successful, and go on to pick and place
        if iid in visible_ids:
            self.curr_state["act"] = PP
            self.measures["success_look_at"] += 1
            return None

        # if num steps for looking is exceeded, fail this rearrangement
        if not self.assert_threshold():
            self.measures["fail_look_at"] += 1
            if self.curr_state["target"] == OBJ:
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.curr_rearrangment = None
                self.log(f"couldn't look at {obj_key} on {rec_key}!")
                return None
            elif self.curr_state["target"] == REC:
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.reset_to_next_best_receptacle()
                self.log(f"trying next best receptacle for {obj_key}, failed to look at {rec_key}!")
                return None

        self.track()
        action = self._look_at_pos(cos_eor_sensor, pos)
        return {"action": action}
    
    def atomic_look_at(self, observations, target_obj, target_rec, target_type) -> dict:
        # Comment: The atomic look-at will include finding and looking at an object, assuming it is within the right proximity
        # If the object is not within grab distance, the look-at will fail after a thresholded number of times
        # Input: target id, type (rec or obj)
        # Output: dictionary of format {"action": a, "flag": success/fail}, success if item in view, fail if tries over threshold

        cos_eor_sensor = observations["cos_eor"][0]
        try:
            if target_type == OBJ:
                oi = target_obj
                target_rec = self.objs[oi]["rec"] # object's current receptacle
                ri = self.get_oi_from_obj_key(target_rec, REC)
                obj_key = self.get_value(oi, OBJ, "obj_key")
            elif target_type == REC:
                ri = target_rec
            
            rec_key = self.get_value(ri, REC, "obj_key")
            iid = self.get_value(ri, REC, "iid")
            # get the position of the target rec
            ep_idx = cos_eor_sensor["recs_keys"].index(rec_key)
            pos = cos_eor_sensor["recs_pos"][ep_idx]
        
        except KeyError as e:
            # example failure case: agent gripping object
            self.temp_trackers[LOOK] = 0
            self.log(f"atomic look-at key error {e}")
            return {"action": None, "flag": FAIL}

        # check if obj/rec is within sensor frame
        visible_ids = torch.unique(observations["semantic"][0]).tolist()
        if iid in visible_ids:
            self.log(f"successful look at {rec_key}")
            self.measures["success_look_at"] += 1
            self.temp_trackers[LOOK] = 0
            return {"action": None, "flag": SUC}

        # if num steps for looking is exceeded, fail this rearrangement
        if self.temp_trackers[LOOK] > self.fail_thresholds[f"{LOOK}-{REC}-{INP}"]:
            self.measures["fail_look_at"] += 1
            self.temp_trackers[LOOK] = 0
            if self.curr_state["target"] == OBJ:
                self.log(f"couldn't look at {obj_key} on {rec_key}!")
                return {"action": None, "flag": FAIL}
            elif self.curr_state["target"] == REC:
                self.log(f"failed to look at {rec_key}!")
                return {"action": None, "flag": FAIL}

        self.track()
        action = self._look_at_pos(cos_eor_sensor, pos)
        self.temp_trackers[LOOK] += 1
        return {"action": action, "flag": INP}

    def pick_place(self, observations):
        gripped_object_id = observations["cos_eor"][0]["gripped_object_id"]
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        cbi = ssi[0]  # current-best index

        obj_id = self.get_value(oi, OBJ, "obj_id") # current rearrangement target object
        ok = self.get_value(oi, OBJ, "obj_key") # object key
        rk = self.get_value(cbi, REC, "obj_key") # rec key

        if self.curr_state["target"] == OBJ:
            iid = self.get_value(oi, OBJ, "iid")
            # picked correct object, navigate to rec
            if obj_id == gripped_object_id:
                self.curr_state["status"] = SUC
                self.track()
                self.curr_state["act"] = NAV
                self.curr_state["status"] = INP
                self.curr_state["target"] = REC
                self.log(f"picked {ok}!")
                self.turn_measures["pick_place_objs"] = f"pick,{ok}"
                return None
            # move-on if failed to pick and tries > threshold
            if not self.assert_threshold():
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.curr_rearrangment = None
                self.log(f"couldn't pick {ok}!")
                return None
            # failed pick attempt, increase counter
            elif gripped_object_id == -1:
                self.track()
                return {"action": 6, "action_args": {"iid": iid}}
            # oracle agent can't pick wrong object
            else:
                raise ValueError

        elif self.curr_state["target"] == REC:
            iid = self.get_value(cbi, REC, "iid")
            # placed on correct receptacle, move-on
            if gripped_object_id == -1:
                self.curr_state["status"] = SUC
                self.track()
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, SUC)
                self.curr_rearrangment = None
                self.objs[oi]["rec"] = rk
                self.log(f"placed {ok} on {rk}!")
                self.turn_measures["pick_place_objs"] = f"place,{ok},{rk}"
                return None
            # move-on to next best receptacle if failed to place and tries > threshold
            elif not self.assert_threshold():
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.reset_to_next_best_receptacle()
                self.log(f"trying next best receptacle for {ok}, failed to place on {rk}!")
                return None
            # try placing

            elif gripped_object_id == obj_id:
                self.track()
                return {"action": 6, "action_args": {"iid": iid}}
            else:
                raise ValueError
            
        
    def atomic_pick_place(self, observations, target_obj, target_rec, target_type):
        # Input: target is an object or rec, stored in target_type
        # the command issued should look like either of the following:
        # - place book (on bookshelf)
        # - pick book (from bookshelf)
        # if the agent chooses pick twice, don't put the item down
        # if target is object -> success if gripped object is the target
        # if target is rec -> success if gripped object put on the rec
        
        try:
            assert(target_obj is not None)
        
        except AssertionError as e:
            # example failure case: agent trying to pick up rec
            self.temp_trackers[PP] = 0
            self.log(f"atomic pick-place assertion error, e.g., pick-up rec {e}")
            return {"action": None, "flag": FAIL}
        
        oi = target_obj
        obj_id = self.get_value(oi, OBJ, "obj_id")
        obj_iid = self.get_value(oi, OBJ, "iid")
        obj_key = self.get_value(oi, OBJ, "obj_key")
        # for picking up the target object
        gripped_object_id = observations["cos_eor"][0]["gripped_object_id"]
        visible_ids = torch.unique(observations["semantic"][0]).tolist()
        current_mapping = observations["cos_eor"][0]["current_mapping"]
            
        if target_type == OBJ:
            # picked correct object, navigate to rec
            if obj_id == gripped_object_id:
                self.objs[oi]["rec"] = None # gripped by the agent
                self.temp_trackers[PP] = 0
                self.log(f"picked {obj_key}!")
                self.turn_measures["pick_place_objs"] = f"pick,{obj_key}"
                return {"action": None, "flag": SUC}
            # move-on if failed to pick and tries > threshold
            elif self.temp_trackers[PP] > self.fail_thresholds[f"{PP}-{OBJ}-{INP}"]:
                self.temp_trackers[PP] = 0
                self.log(f"couldn't pick {obj_key}!")
                return {"action": None, "flag": FAIL}
            # failed pick attempt, increase counter
            elif gripped_object_id == -1 or gripped_object_id != obj_id:
                self.temp_trackers[PP] += 1
                return {"action": 6, "action_args": {"iid": obj_iid}, "flag": INP}
            else:
                # one failure case: gripping something else
                self.temp_trackers[PP] += 1
                return {"action": 6, "action_args": {"iid": obj_iid}, "flag": INP}
                # raise ValueError
            
        elif target_type == REC:
            ri = target_rec
            # for placing an item on the target receptacle
            rec_iid = self.get_value(ri, REC, "iid")
            # the key of the target rec to place or the rec to pick up the object
            rec_key = self.get_value(ri, REC, "obj_key")
            # placed on correct receptacle, move-on
            if gripped_object_id == -1: # and current_mapping[obj_key] == rec_key:
                self.temp_trackers[PP] = 0
                self.objs[oi]["rec"] = rec_key
                if current_mapping[obj_key] == rec_key:
                    self.log(f"placed {obj_key} on {rec_key}!")
                else:
                    self.log(f"did not placed {obj_key} on {rec_key}!")
                self.turn_measures["pick_place_objs"] = f"place,{obj_key},{rec_key}"
                return {"action": None, "flag": SUC }
            # move-on to next best receptacle if failed to place and tries > threshold
            elif self.temp_trackers[PP] > self.fail_thresholds[f"{PP}-{OBJ}-{INP}"]:
                self.temp_trackers[PP] = 0
                self.log(f"failed to place {obj_key} on {rec_key}!")
                return {"action": None, "flag": FAIL}
            # try placing
            elif gripped_object_id == obj_id:# and rec_iid in visible_ids:
                self.temp_trackers[PP] += 1
                return {"action": 6, "action_args": {"iid": rec_iid}, "flag": INP}
            else:
                # one failure case: rec not visible
                self.temp_trackers[PP] += 1
                return {"action": 6, "action_args": {"iid": rec_iid}, "flag": INP}
                #raise ValueError

    def reset_to_next_best_receptacle(self):
        oi, cmi, (ssi, ss) = self.curr_rearrangment

        # remove current best receptacle
        ssi, ss = ssi[1:], ss[1:]

        self.curr_rearrangment = oi, cmi, (ssi, ss)
        # reset state to navigate
        self.curr_state["status"] = INP
        self.curr_state["target"] = REC
        self.curr_state["act"] = NAV

    def log(self, text):
        file = "oracle-log.txt"
        with open(file, 'a') as f:
            print(text, file=f)
        print(text)

    def in_view(self, obs, oi):
        iid = self.get_value(oi, OBJ, "obj_key")
        avail_sids = obs["sid"].squeeze().unique()

    def fail_nav(self):
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        self.curr_state["status"] = FAIL
        self.track()
        self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
        self.curr_rearrangment = None

    def _get_agent_obj_dist(self, obj_idx, obj_type, dist_type):
        obj_id = self.get_value(obj_idx, obj_type, "obj_id")
        return self.envs.call_at(0, "get_agent_object_distance", {"obj_id": obj_id, "dist_type": dist_type})

    def _get_obj_obj_dist(self, obj1_idx, obj1_type, obj2_idx, obj2_type, dist_type):
        obj1_id = self.get_value(obj1_idx, obj1_type, "obj_id")
        obj2_id = self.get_value(obj2_idx, obj2_type, "obj_id")
        return self.envs.call_at(
            0,
            "get_object_object_distance",
            {"obj1_id": obj1_id, "obj2_id": obj2_id, "dist_type": dist_type}
        )

    def nav(self, observations):
        gripped_object_id = observations["cos_eor"][0]["gripped_object_id"]
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        cbi = ssi[0]
        grab_dist = self.task_params.ACTIONS.GRAB_RELEASE.GRAB_DISTANCE
        if self.curr_state["target"] == OBJ:
            agent_rec_dist = self._get_agent_obj_dist(cmi, REC, "l2")
            ok = self.get_value(oi, OBJ, "obj_key")
            rk = self.get_value(cmi, REC, "obj_key")
            # move-on if failed to threshol navigate
            if not self.assert_threshold():
                self.fail_nav()
                self.log(f"can't reach {ok} on {rk}! exceeded step threshold")
                return None
            else:
                # take a step towards goal
                num_nav_steps = self.track(only_return=True)
                targets = {
                    "nav": NavTargetInfo(REC, cmi),
                    "look": NavTargetInfo(REC, cmi)
                }
                _, action, _, _, module, _ = self.nav_module.act(observations, targets, self, num_nav_steps)
                if num_nav_steps == 0:
                    print(f"snap to {ok} on {rk}")
                self.track()
                # navigation completed
                if action == 0:
                    # check for success, and move to picking
                    if agent_rec_dist < grab_dist:
                        self.curr_state["status"] = SUC
                        self.track()
                        self.curr_state["act"] = LOOK
                        self.curr_state["status"] = INP
                        self.log(f"reached {ok} on {rk}!")
                        return None
                    # failed to navigate
                    else:
                        self.fail_nav()
                        self.log(
                            f"can't reach {ok} on {rk}! "
                            f"pathplanner failed with dist {agent_rec_dist} after {num_nav_steps} steps"
                        )
                        return None
                return {"action": action}

        elif self.curr_state["target"] == REC:
            assert gripped_object_id != -1
            agent_rec_dist = self._get_agent_obj_dist(cbi, REC, "l2")
            rk = self.get_value(cbi, REC, "obj_key")
            # try next best receptacle if failed
            if not self.assert_threshold():
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.reset_to_next_best_receptacle()
                self.log(f"trying next best receptacle, failed to reach {rk}! exceeded step threshold")
                return None
            # take a step towards goal
            else:
                num_nav_steps = self.track(only_return=True)
                targets = {
                    "nav": NavTargetInfo(REC, cbi),
                    "look": NavTargetInfo(REC, cbi)
                }
                _, action, _, _, module, _ = self.nav_module.act(observations, targets, self, num_nav_steps)
                self.track()
                if action == 0:
                    # check success, and move to placing
                    if agent_rec_dist < grab_dist:
                        self.curr_state["status"] = SUC
                        self.track()
                        self.curr_state["act"] = LOOK
                        self.curr_state["status"] = INP
                        self.log(f"reached {rk}!")
                        return None
                    else:
                        self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                        self.reset_to_next_best_receptacle()
                        self.log(
                            f"trying next best receptacle, failed to reach {rk}! "
                            f"pathplanner failed with dist {agent_rec_dist} after {num_nav_steps} steps"
                        )
                        return None
                return {"action": action}
        else:
            raise AssertionError
    
    
    def atomic_nav(self, observations, target_obj, target_rec, target_type):
        # atomic navigation with input of target object or target rec, target room not yet implemented
        # Alternatively, we may search for a target rec in that room and navigate
        
        grab_dist = self.task_params.ACTIONS.GRAB_RELEASE.GRAB_DISTANCE
        
        if target_type == OBJ:
            try:
                oi = target_obj
                target_rec = self.objs[oi]["rec"]
                ri = self.get_oi_from_obj_key(target_rec, REC)
                agent_rec_dist = self._get_agent_obj_dist(ri, REC, "l2")
                ok = self.get_value(oi, OBJ, "obj_key")
                rk = self.get_value(ri, REC, "obj_key")
                # move-on if failed to threshold navigate
            except KeyError as e:
                # example failure case: agent gripping object
                self.temp_trackers[NAV] = 0
                self.log(f"atomic nav key error {e}")
                return {"action": None, "flag": FAIL}
            if self.temp_trackers[NAV] > self.fail_thresholds[f"{NAV}-{OBJ}-{INP}"]:
                self.temp_trackers[NAV] = 0
                self.log(f"can't reach {ok} on {rk}! exceeded step threshold")
                return {"action": None, "flag": FAIL}
            else:
                # take a step towards goal
                nav_targets = {
                    "nav": NavTargetInfo(REC, ri),
                    "look": NavTargetInfo(REC, ri)
                }
                try:
                    _, action, _, _, module, _ = self.nav_module.act(observations, nav_targets, self, self.temp_trackers[NAV])
                except KeyError as e:
                    self.log(f"nav error {e}")
                    self.temp_trackers[NAV] = 0
                    return {"action": None, "flag": FAIL}
                if self.temp_trackers[NAV] == 0:
                    print(f"snap to {ok} on {rk}")
                self.track()
                # navigation completed
                if action == 0:
                    # check for success
                    if agent_rec_dist < grab_dist:
                        self.log(f"reached {ok} on {rk} in {self.temp_trackers[NAV]} steps!")
                        self.temp_trackers[NAV] = 0
                        return {"action": None, "flag": SUC}
                    # failed to navigate
                    else:
                        self.log(
                            f"can't reach {ok} on {rk}! "
                            f"pathplanner failed with dist {agent_rec_dist} after {self.temp_trackers[NAV]} steps"
                        )
                        self.temp_trackers[NAV] = 0
                        return {"action": None, "flag": FAIL}
                else:
                    self.temp_trackers[NAV] += 1
                return {"action": action, "flag": INP}

        elif target_type == REC:
            try:
                ri = target_rec
                agent_rec_dist = self._get_agent_obj_dist(ri, REC, "l2")
                rk = self.get_value(ri, REC, "obj_key")
            except KeyError as e:
                # example failure case: agent gripping object
                self.temp_trackers[NAV] = 0
                self.log(f"atomic nav key error {e}")
                return {"action": None, "flag": FAIL}
            # try next best receptacle if failed
            if self.temp_trackers[NAV] > self.fail_thresholds[f"{NAV}-{REC}-{INP}"]:
                self.log(f"failed to reach {rk}! exceeded step threshold")
                return {"action": None, "flag": FAIL}
            # otherwise, take a step towards goal
            else:
                nav_targets = {
                    "nav": NavTargetInfo(REC, ri),
                    "look": NavTargetInfo(REC, ri)
                }
                try:
                    _, action, _, _, module, _ = self.nav_module.act(observations, nav_targets, self, self.temp_trackers[NAV])
                except KeyError as e:
                    self.log(f"nav error {e}")
                    self.temp_trackers[NAV] = 0
                    return {"action": None, "flag": FAIL}
                if action == 0:
                    # check success, and move to placing
                    if agent_rec_dist < grab_dist:
                        self.log(f"reached {rk} in {self.temp_trackers[NAV]} steps!")
                        self.temp_trackers[NAV] = 0
                        return {"action": None, "flag": SUC}
                    else:
                        self.log(
                            f"trying next best receptacle, failed to reach {rk}! "
                            f"pathplanner failed with dist {agent_rec_dist} after {self.temp_trackers[NAV]} steps"
                        )
                        self.temp_trackers[NAV] = 0
                        return {"action": None, "flag": FAIL}
                else:
                    self.temp_trackers[NAV] += 1
                return {"action": action, "flag": INP}
        elif target_type == PRIMITIVE:
            # if first call, execute the action and return in-progress
            # in second call, return done action and return SUC of FAIL
            # call the primitive action
            if self.temp_trackers[NAV] == 0:
                action = target_obj
                self.temp_trackers[NAV] += 1
                return {"action": action, "flag": INP}
            else:
                action = None
                self.temp_trackers[NAV] = 0 
                return {"action": action, "flag": SUC}
        else:
            raise AssertionError


    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        pass

    def get_seen(self, attr, types=["rec", "obj"]):
        """get specified attributes of already seen objects/receptacles"""
        attr_list = []
        if "rec" in types:
            attr_list.extend([ed[attr] for ed in self.rec_rooms.values()])
        if "obj" in types:
            attr_list.extend([ed[attr] for ed in self.objs.values()])
        return attr_list
    
    def get_obj_id_from_obj_key(self, obj_key):
        if not obj_key is None:
            return self.obj_key_to_sim_obj_id[obj_key]
        
        
    def get_oi_from_obj_key(self, obj_key, type):
        if not obj_key is None:
            if type == REC:
                for oi in self.rec_rooms:
                    if self.rec_rooms[oi]["obj_key"] == obj_key:
                        return oi
            elif type == OBJ:
                for oi in self.objs:
                    if self.objs[oi]["obj_key"] == obj_key:
                        return oi
            raise KeyError(f"rec {obj_key} not found")
        

    def get_value(self, idx, type, key):
        if type == REC:
            # floor
            if idx == -1:
                if key == "obj_key":
                    return "floor"
                else:
                    raise ValueError
            return self.rec_rooms[idx][key]
        elif type == OBJ:
            return self.objs[idx][key]
        else:
            raise ValueError

    def debug(self, task_data, visible_iids):
        obj_keys = [
            "laptop_7_0.urdf",
            "table_9_0.urdf",
            "coffee_table_5_0.urdf",
            "013_apple_1",
            "026_sponge_2",
            "counter_26_0.urdf",
            "sink_35_0.urdf",
            "sink_42_0.urdf"
        ]
        visible_iids += [task_data["sim_obj_id_to_iid"][task_data['obj_key_to_sim_obj_id'][ok]] for ok in obj_keys]
        visible_iids = list(set(visible_iids))
        return visible_iids

    def update(self, obs):
        task_data = obs["cos_eor"][0]
        if self.policy_params.explore.type == "oracle":
            visible_iids = task_data["iid_to_sim_obj_id"].keys()
        else:
            # currently visible iids
            visible_iids = obs["semantic"][0].unique().tolist()
        cur_visible_objs = [task_data["sim_obj_id_to_obj_key"][task_data["iid_to_sim_obj_id"][v]] for v in obs['visible_obj_iids'][0].unique().tolist() if v != 0]
        cur_visible_recs = [task_data["sim_obj_id_to_obj_key"][task_data["iid_to_sim_obj_id"][v]] for v in obs['visible_rec_iids'][0].unique().tolist() if v != 0]
        gripped_object_id = task_data["gripped_object_id"]
        if gripped_object_id == -1:
            gripped_object_key = NOTHING
        else:
            gripped_object_key = task_data["sim_obj_id_to_obj_key"].get(gripped_object_id)
        task_data[CUR_VISIBLE_OBJ_KEYS] = cur_visible_objs
        task_data[CUR_VISIBLE_REC_KEYS] = [k for k in cur_visible_recs if not any(w in k for w in ["None-", "picture", "door", "window"])]
        task_data[GRIPPED_OBJ_KEY] = gripped_object_key
        
        if self.key_translator.simplify_keys:
            all_keys = [task_data["sim_obj_id_to_obj_key"][task_data["iid_to_sim_obj_id"][v]] for v in task_data["iid_to_sim_obj_id"].keys()]
            self.key_translator.update(all_keys)

        if len(self.room_recs) == 0:
            all_item_ids = [task_data["iid_to_sim_obj_id"][k] for k in task_data["iid_to_sim_obj_id"].keys()] # paired with all_keys
            all_keys = [task_data["sim_obj_id_to_obj_key"][v] for v in all_item_ids]
            for index, obj_id in enumerate(all_item_ids):
                obj_key = all_keys[index]
                room = task_data["obj_id_to_room"][obj_id]
                if 'urdf' in obj_key and room != 'null':
                    if not room in self.room_recs:
                        self.room_recs[room] = [obj_key]
                    else:
                        self.room_recs[room].append(obj_key)
            self.key_translator.add_rooms(self.room_recs)
        task_data[ROOM] = list(self.key_translator.rooms.keys())

                
        if 0 in visible_iids:
            visible_iids.remove(0)
        visible_obj_ids = [task_data["iid_to_sim_obj_id"][iid] for iid in visible_iids]
        visible_obj_keys = [task_data["sim_obj_id_to_obj_key"][oid] for oid in visible_obj_ids]
        obj_iid_to_idx = {obj["iid"]: idx for idx, obj in self.objs.items()}
        rec_iid_to_idx = {rr["iid"]: idx for idx, rr in self.rec_rooms.items()}
        novel_objects = [
            vk
            for vk, vi in zip(visible_obj_keys, visible_iids)
            if (vi not in rec_iid_to_idx) and (vi not in obj_iid_to_idx) and ("door" not in vk)
        ]
        self.turn_measures["novel_objects"] = ",".join(novel_objects)
        if len(novel_objects):
            self.log(f"discovered: {novel_objects}")
        for iid, obj_id, obj_key in zip(visible_iids,  visible_obj_ids, visible_obj_keys):
            # floor etc
            if iid == 0:
                continue
            sid = task_data["iid_to_sid"][iid]
            obj_type = task_data["sim_obj_id_to_type"][obj_id]
            entity_dict = {
                "sid": sid,
                "iid": iid,
                "obj_id": obj_id,
                "sem_class": task_data["sid_class_map"][sid],
                "room": task_data["obj_id_to_room"][obj_id],
                "obj_type": obj_type,
                "obj_key": obj_key,
            }

            if obj_type == "rec" and "door" not in obj_key:
                ep_idx = task_data["recs_keys"].index(obj_key)
                entity_dict["pos"] = task_data["recs_pos"][ep_idx]
                entity_dict["objs"] = [o for o,r in task_data["current_mapping"].items() if r == obj_key and r in visible_obj_keys]
                rec_idx = rec_iid_to_idx.get(iid, len(self.rec_rooms))
                if iid in rec_iid_to_idx:
                    # if we have previously seen this receptacle, accumulate objs
                    entity_dict["objs"] = list(set(entity_dict["objs"]) or set(self.rec_rooms[rec_idx]["objs"]))
                self.rec_rooms[rec_idx] = entity_dict
            elif obj_type == "obj":
                ep_idx = task_data["objs_keys"].index(obj_key)
                entity_dict["pos"] = task_data["objs_pos"][ep_idx]
                # map receptacle if it is visible
                entity_dict["rec"] = task_data["current_mapping"][obj_key]
                obj_idx = obj_iid_to_idx.get(iid, len(self.objs))
                self.objs[obj_idx] = entity_dict
            elif "door" not in obj_key:
                raise ValueError

        # update the context generator
        self.context_module.update(task_data)
        if len(self.obj_key_to_sim_obj_id) == 0:
            self.obj_key_to_sim_obj_id = task_data['obj_key_to_sim_obj_id']

        # TODO: support non-interactable object type
        bad_rec_substrs = ["-picture_", "-window_"]
        self.bad_recs = set()
        for idx, rec in self.rec_rooms.items():
            key = rec["obj_key"]
            if any(bad_sub in key for bad_sub in bad_rec_substrs):
                self.bad_recs.add(idx)

        self.assert_consistency()

    def update_pick_place(self, type, oid, rid):
        # update the internal mapping when pick/place occurs
        self.assert_consistency()
        pass

    def get_info(self):
        return self.measures

# translate the key names with underscores to simplified name
class KeyTranslator:
    def __init__(self, simplify_keys=False):
        self.simplify_keys = simplify_keys
        self.rooms = {} # mapping from room name to full key of rec in the room
        self.full_to_simplified_key_map = {} # mapping from full key to simplified key
        self.simplified_to_full_key_map = {} # mapping from simplified key to full key
        self.simplified_rec_to_simplified_room_map = {} # mapping from simplified rec to simplified room
        self.full_rec_to_full_room_map = {} # mapping from full rec to full room
    
    def reset(self):
        self.rooms = {} # mapping from room name to full key of rec in the room
        self.full_to_simplified_key_map = {} # mapping from full key to simplified key
        self.simplified_to_full_key_map = {} # mapping from simplified key to full key
        self.simplified_rec_to_simplified_room_map = {} # mapping from simplified rec to simplified room
        self.full_rec_to_full_room_map = {} # mapping from full rec to full room
        
    def update(self, keys):
        # fill in the key maps
        # the full key may contain upper case, the simplified key only contains lower case
        for k in keys:
            if not k in self.full_to_simplified_key_map:
                if "urdf" in k:
                    # it is a REC, e.g., living_room_0-bottom_cabinet_0_0.urdf
                    # Regular expression to match the pattern
                    match = re.match(r'^(?P<room>[^-]+)-(?P<rec>.+?)_\d+\.urdf$', k.lower())
                    
                    if match:
                        room = match.group('room')
                        self.full_rec_to_full_room_map[k] = room
                        rec = match.group('rec')
                        rec = rec.replace('_no_top', "").replace('_', " ")
                        if room == "none":
                            simplified_k = rec
                        else:
                            room = room.replace('_', " ")
                            simplified_k = f"{room} {rec}"
                        self.simplified_rec_to_simplified_room_map[simplified_k] = room
                    else:
                        print ('regex failed to parse rec', k)
                else:
                    # it is an OBJ, e.g., softball_1
                    simplified_k = k.strip().replace('_', " ")
                self.full_to_simplified_key_map[k] = simplified_k
                self.simplified_to_full_key_map[simplified_k] = k
                
    def get_room_from_rec(self, rec):
        if self.simplify_keys:
            return self.simplified_rec_to_simplified_room_map[rec]
        else:
            return self.full_rec_to_full_room_map[rec]
    
    def add_rooms(self, room_rec: dict):
        # only add the room -> rec in the simplified to full key map
        for room in room_rec:
            if self.simplify_keys:
                self.rooms[room.replace('_', ' ')] = room_rec[room][0]
            else:
                self.rooms[room] = room_rec[room][0]

    def is_room(self, key):
        return key in self.rooms
    
    def translate(self, key, simplify=True):
        # from full keys or from simplified keys, by default, translate from full key to simplified key
        if key.lower() == AGENT:
            return {VAL: key, FLAG: True}
        if key in self.rooms:
            if simplify:
                return {VAL: key, FLAG: True}
            else:
                return {VAL: self.rooms[key], FLAG: True}
        if simplify:
            if key in self.full_to_simplified_key_map:
                result = {VAL:self.full_to_simplified_key_map[key], FLAG: True}
            else:
                result = {VAL: key, FLAG: False}
        else:
            if key in self.simplified_to_full_key_map:
                result = {VAL: self.simplified_to_full_key_map[key], FLAG: True}
            else:
                result = {VAL:key, FLAG: False}
        return result
    
    
class HouseLogger:
    def __init__(self, plan_mode=SINGLE):
        self.records = []
        self.record = None
        self.log = None
        self.plan_mode = plan_mode # SINGLE or ADAPTER
        self.timestep = 0
    
    def reset(self):
        self.records = []
        self.timestep = 0
        self.record = None
        self.log = None
        self.log_open = False
    
    def get_prev_steps(self):
        low_level_steps = []
        high_level_steps = []
        low_level_steps_suc = []
        for record in self.records:
            high_level_steps.append(record[HIGH_LEVEL][STEP])
            for log in record[LOGS]:
                low_level_steps.append(log[STEP_RAW])
                if log[FLAG] == SUC:
                    low_level_steps_suc.append(log[STEP_RAW])
        if self.record is not None:
            # append the current record information
            if HIGH_LEVEL in self.record:
                high_level_steps.append(self.record[HIGH_LEVEL][STEP])
            if LOGS in self.record:
                for log in self.record[LOGS]:
                    low_level_steps.append(log[STEP_RAW])
                    if log[FLAG] == SUC:
                        low_level_steps_suc.append(log[STEP_RAW])
        return {HIGH_LEVEL:{ALL: high_level_steps}, LOW_LEVEL:{ALL: low_level_steps, SUC: low_level_steps_suc}}
        
        
    # adding a new record when prompt and plan
    def create_record(self, high_level_prompt, low_level_prompt, high_level_text, low_level_text, high_level_step):
        self.record = {}
        self.record[MODE] = self.plan_mode
        self.record[START] = self.timestep
        self.record[HIGH_LEVEL] = {PROMPT: high_level_prompt, RESPONSE: high_level_text, STEP: high_level_step}
        self.record[LOW_LEVEL] = {PROMPT: low_level_prompt, RESPONSE: low_level_text}
        # log a record when prompt and plan
        self.record[LOGS] = []
        
    # Step 1 of two-step process to make a step log
    def open_execution_record(self, step_raw, step_parsed, current_mapping, correct_mapping):
        # take a snapshot of the graph
        self.log = {}
        self.log[START] = self.timestep
        self.log[STEP_RAW] = step_raw
        self.log[STEP_PARSED] = step_parsed
        self.log[CURRENT_MAPPING] = {START: copy.deepcopy(current_mapping)}
        self.log[CORRECT_MAPPING] = copy.deepcopy(correct_mapping)
        self.log_open = True
    
    # Step 2 of two-step process to make a step log
    def close_execution_record(self, flag, discovered_objects, discovered_recs, moved_objects, current_mapping):
        # summarize the changes in the graph
        self.log[END] = self.timestep
        self.log[FLAG] = flag
        self.log[CURRENT_MAPPING][END] = copy.deepcopy(current_mapping)
        correct_placed = {START: 0, END: 0}
        wrong_placed = {START: 0, END: 0}
        for w in [START, END]:
            for obj in self.log[CURRENT_MAPPING][w]:
                rec = self.log[CURRENT_MAPPING][w][obj]
                if rec in self.log[CORRECT_MAPPING][obj]:
                    correct_placed[w] += 1
                else:
                    wrong_placed[w] += 1
        self.log[OUTCOME] = {OBJ_DISCOVERED: discovered_objects, 
                             REC_DISCOVERED: discovered_recs, 
                             OBJ_MOVED: moved_objects,
                             COUNT_CORRECT: correct_placed,
                             COUNT_WRONG: wrong_placed
                             }
        self.record[LOGS].append(self.log)
        self.log_open = False
    
    # close the record when the plan has finished execution
    # Specifically, called when new prompt and plan, and when whole plan finishes.
    def summarise_record(self):
        if self.record is None:
            return
        if len(self.record[LOGS]) == 0:
            return 
        discovered_objects = set()
        discovered_recs = set()
        moved_objects = {}
        self.record[CORRECT_MAPPING] = copy.deepcopy(self.record[LOGS][0][CORRECT_MAPPING])
        for log in self.record[LOGS]:
            discovered_objects.update(set(log[OUTCOME][OBJ_DISCOVERED]))
            discovered_recs.update(set(log[OUTCOME][REC_DISCOVERED]))
            for obj in log[OUTCOME][OBJ_MOVED]:
                if obj in moved_objects:
                    # update the target
                    moved_objects[obj] = (moved_objects[obj][0], log[OUTCOME][OBJ_MOVED][obj][1])
                else:
                    moved_objects[obj] = log[OUTCOME][OBJ_MOVED][obj]
            log.pop(CORRECT_MAPPING, None)
        self.record[CURRENT_MAPPING] = {START: self.record[LOGS][0][CURRENT_MAPPING][START], END: self.record[LOGS][-1][CURRENT_MAPPING][END]}
        # find correctly placed objects start and end
        self.record[CORRECT_OBJ] = {START: {}, END: {}}
        self.record[WRONG_OBJ] = {START: {}, END: {}}
        for w in [START, END]:
            for obj in self.record[CURRENT_MAPPING][w]:
                rec = self.record[CURRENT_MAPPING][w][obj]
                if rec in self.record[CORRECT_MAPPING][obj]:
                    self.record[CORRECT_OBJ][w][obj] = rec
                else:
                    self.record[WRONG_OBJ][w][obj] = rec
        self.record[OUTCOME] = {OBJ_DISCOVERED: list(discovered_objects), 
                                REC_DISCOVERED: list(discovered_recs), 
                                OBJ_MOVED: moved_objects,
                                COUNT_CORRECT: {START: len(self.record[CORRECT_OBJ][START]), END: len(self.record[CORRECT_OBJ][END])},
                                COUNT_WRONG: {START: len(self.record[WRONG_OBJ][START]), END: len(self.record[WRONG_OBJ][END])}
                                }
        self.record[END] = self.timestep
        self.records.append(self.record)
        
    def write_log_to_file(self, output_filepath):
        with open(output_filepath, 'w') as file:
            json.dump(self.records, file, indent=4)
        print (f"log written to {output_filepath}")