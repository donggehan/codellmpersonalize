# A class which generates description of the global context information for the LLM planners

from cos_eor.utils.constants import *
from collections import OrderedDict
import json


class ContextModule:
    def __init__(self, config, num_envs):
        self.config = config
        self.num_envs = num_envs
        self.nl_context = ""
        self.system_context = self.config[SINGLE][SYSTEM_PROMPT]
        self.simplify_keys = self.config.get(SIMPLIFY_KEYS, False) # by default, don't simplify keys
        self.context_obs_dict = {}
        self.house_graph = HouseGraph(self.config[GRAPH])
        # global info for reward generation
        self.current_mapping = {}
        self.correct_mapping = {}
        
    def reset(self, key_translator=None, house_logger=None):
        self.nl_context = ""
        self.key_translator = key_translator
        self.house_logger = house_logger
        self.house_graph.reset()
        # global info for reward generation
        self.current_mapping = {} # updated in update_graph
        self.correct_mapping = {} # updated in update_graph
        

    def update(self, context_obs_dict):
        """this is called every single step"""
        self.context_obs_dict = context_obs_dict
        self.update_graph()
        self.nl_context = self._format_prompt(
            self.config[SINGLE][BASE_PROMPT], context_obs_dict, which_model=SINGLE
        )
        
    def update_graph(self):
        # Note: called after context_obs_dict is updated
        # rooms: ['kitchen 1', 'bedroom 1', ...]
        rooms = list(self.key_translator.rooms.keys())
        visible_objects = self.context_obs_dict[CUR_VISIBLE_OBJ_KEYS]
        visible_recs = self.context_obs_dict[CUR_VISIBLE_REC_KEYS]
        gripped_obj = self.context_obs_dict[GRIPPED_OBJ_KEY]
        # for reward calculation
        self.correct_mapping = self.context_obs_dict[CORRECT_MAPPING]
        self.current_mapping = self.context_obs_dict[CURRENT_MAPPING]

        if self.simplify_keys:
            visible_objects = [self.key_translator.translate(k, simplify=True)[VAL] for k in visible_objects]
            visible_recs = [self.key_translator.translate(k, simplify=True)[VAL] for k in visible_recs]
            gripped_obj = self.key_translator.translate(gripped_obj, simplify=True)[VAL]
            # for reward calculation
            translated_correct_mapping = {}
            for k in self.correct_mapping:
                vals = self.correct_mapping[k]
                translated_vals = [self.key_translator.translate(v, simplify=True)[VAL] for v in vals]
                translated_correct_mapping[self.key_translator.translate(k, simplify=True)[VAL]] = translated_vals
            self.correct_mapping = translated_correct_mapping
            self.current_mapping = {self.key_translator.translate(k, simplify=True)[VAL]:self.key_translator.translate(self.current_mapping[k], simplify=True)[VAL] for k in self.current_mapping}
        # now convert them to the output format
        # visible_objects: {'laptop 1':{"room": "kitchen 1", "rec": "kitchen 1 sink 2"}, ...}
        # visible_recs: {"kitchen 1 sink 2": "kitchen 1", ...}
        visible_recs = {k: self.key_translator.get_room_from_rec(k) for k in visible_recs}
        temp_visible_objects= {}
        for obj in [gripped_obj, self.house_graph.gripped_obj]:
            # the agent can observe the obj it's currently holding and the obj it just placed
            if obj != NOTHING:
                visible_objects = visible_objects + [obj]
        for obj in visible_objects:
            rec = self.current_mapping[obj]
            if rec == AGENT:
                temp_visible_objects[obj] = {ROOM: None, REC: rec}
            else:
                temp_visible_objects[obj] = {ROOM: self.key_translator.get_room_from_rec(rec), REC: rec}
        
        self.house_graph.update(rooms, temp_visible_objects, visible_recs, gripped_obj)
        
    def snapshot_house_graph(self):
        self.house_graph.snapshot()
        
    def diff_house_graph(self):
        return self.house_graph.diff()
    
    def get_observed(self):
        return self.house_graph.get_observed()

    def get_nl_context(self) -> str:
        return {USER: self.nl_context, SYSTEM: self.system_context}

    def get_debug_info(self) -> dict:
        correct_mapping = self.context_obs_dict[CORRECT_MAPPING]
        current_mapping = self.context_obs_dict[CURRENT_MAPPING]
        rooms = self.context_obs_dict[ROOM]
        if self.key_translator.simplify_keys:
            translated_correct_mapping = {}
            for k in correct_mapping:
                vals = correct_mapping[k]
                translated_vals = [self.key_translator.translate(v, simplify=True)[VAL] for v in vals]
                translated_correct_mapping[self.key_translator.translate(k, simplify=True)[VAL]] = translated_vals
            correct_mapping = translated_correct_mapping
            current_mapping = {self.key_translator.translate(k, simplify=True)[VAL]:self.key_translator.translate(current_mapping[k], simplify=True)[VAL] for k in current_mapping}
        return {CORRECT_MAPPING: correct_mapping, CURRENT_MAPPING: current_mapping, ROOM: rooms, GRAPH: self.house_graph, ALL: self.context_obs_dict}

    # helper functions
    def _format_prompt(self, base_prompt: str, context_obs_dict, which_model) -> str:
        cur_prompt = base_prompt
        # action placeholder
        self.option_list = self.config[which_model].get(OPTION_LIST, '') # placeholder only in the adapter
        if self.allowed_options[ROOM]:
            self.option_list = 'go to room, ' + self.option_list
        if self.allowed_options[PRIMITIVE]:
            self.option_list += ', ' + ','.join([MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN])
        cur_prompt = cur_prompt.replace(OPTION_PLACEHOLDER, self.option_list)
        placeholder_texts = []
        placeholder_examples = []
        for key in self.config[which_model][PLANNER_CONTEXT_OBS_KEYS]:
            if key == EXAMPLE_EXPLORE:
                nl = self.config[which_model][EXAMPLE_EXPLORE]
                placeholder_examples.append(nl)
            elif key == EXAMPLE_PICK:
                nl = self.config[which_model][EXAMPLE_PICK]
                placeholder_examples.append(nl)
            elif key == GRAPH:
                nl = self.house_graph.convert_graph_to_text()
                placeholder_texts.append(nl)
            else:
                nl = self._translate(context_obs_dict[key], key)
                placeholder_texts.append(nl)
        cur_prompt = cur_prompt.replace(CONTEXT_PLACEHOLDER, ".".join(placeholder_texts))
        cur_prompt = cur_prompt.replace(EXAMPLE_PLACEHOLDER, "".join(placeholder_examples) + '\n')
        return cur_prompt

    def _translate(self, obj, key) -> str:
        # before converting, first simplify the keys using the key_translator
        obj_translated = None
        if not self.key_translator.simplify_keys:
            obj_translated = obj
        else:
            if type(obj) == dict:
                for k in obj:
                    v = obj[k]
                    obj_translated[self.key_translator.translate(k, simplify=True)[VAL]] = self.key_translator.translate(v, simplify=True)[VAL]
            elif type(obj) == list:
                obj_translated = [self.key_translator.translate(k, simplify=True)[VAL] for k in obj]
            elif type(obj) == str:
                obj_translated = self.key_translator.translate(obj, simplify=True)[VAL]
        obj = obj_translated
        # convert the object to natural language   
        if key == RECS_KEYS:
            # input recs are a list
            # remove the rec keys which do not make sense
            recs = [k for k in obj if AGENT not in k]
            nl = f"A list of receptacles are {','.join(recs)}"
        
        elif key == CURRENT_MAPPING:
            # input mapping are ordered dictionary
            nl = ", ".join(
                f"{tup[0]} is on {tup[1]}" if tup[1] != AGENT else f"You are gripping {tup[0]}" for tup in obj.items()
            )
        elif key == CUR_VISIBLE_OBJ_KEYS:
            nl = f"A list of visible objects are {','.join(obj)}"
        elif key == CUR_VISIBLE_REC_KEYS:
            nl = f"A list of visible receptacles are {','.join(obj)}"
        elif key == GRIPPED_OBJ_KEY:
            nl = f"\nYou are holding {obj}"
        elif key == ROOM:
            num_rooms = len(obj)
            rooms_str = ','.join(obj)
            nl = f"There are {num_rooms} ROOM in this house: {rooms_str}"
        else:
            print ( f"error retrieving {key} for context generator" )
            
        return nl

# context module which provides a prompt as a dictionary to both planner and adapter
class PlannerAdapterContextModule(ContextModule):
    def __init__(self, config, num_envs):
        self.config = config
        self.num_envs = num_envs
        self.nl_context = {}
        self.simplify_keys = self.config.get(SIMPLIFY_KEYS, False) # by default, don't simplify keys
        self.system_context_planner = self.config[HIGH_LEVEL][SYSTEM_PROMPT]
        self.system_context_adapter = self.config[ADAPTER][SYSTEM_PROMPT]
        self.context_obs_dict = {}
        self.house_graph = HouseGraph(self.config[GRAPH])
        self.current_mapping = {}
        self.correct_mapping = {}
        
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)

    def update(self, context_obs_dict):
        """this is called every single step"""
        self.context_obs_dict = context_obs_dict
        self.update_graph()
        nl_context_planner = self._format_prompt(
            self.config[HIGH_LEVEL][BASE_PROMPT], context_obs_dict, which_model=HIGH_LEVEL
        )
        nl_context_adapter = context_obs_dict
        nl_context_adapter = self._format_prompt(
            self.config[ADAPTER][BASE_PROMPT], context_obs_dict, which_model=ADAPTER
        )
        self.nl_context = {PLANNER: nl_context_planner, ADAPTER: nl_context_adapter}

    def get_nl_context(self) -> str:
        result = {
            PLANNER: {USER: self.nl_context[PLANNER], SYSTEM: self.system_context_planner},
            ADAPTER: {USER: self.nl_context[ADAPTER], SYSTEM: self.system_context_adapter}
        }
        return result
    
    def get_debug_info(self) -> dict:
        return self.context_obs_dict
    
    
    
class SayPlanContextModule(ContextModule):
    def __init__(self, config, num_envs):
        self.config = config
        self.num_envs = num_envs
        self.nl_context = {}
        self.simplify_keys = self.config.get(SIMPLIFY_KEYS, False) # by default, don't simplify keys
        self.system_context_search = self.config[SAYPLAN][SYSTEM_PROMPT_SEARCH]
        self.system_context_planner = self.config[SAYPLAN][SYSTEM_PROMPT]
        self.context_obs_dict = {}
        self.house_graph = HouseGraph(self.config[GRAPH])
        self.current_mapping = {}
        self.correct_mapping = {}
        
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)

    def update(self, context_obs_dict):
        """this is called every single step"""
        self.context_obs_dict = context_obs_dict
        self.update_graph()
        nl_context_search = self.config[SAYPLAN][BASE_PROMPT_SEARCH]
        nl_context_planner = self._format_prompt(
            self.config[SAYPLAN][BASE_PROMPT], context_obs_dict, which_model=SAYPLAN
        )
        self.nl_context = {PLANNER: nl_context_planner, SEARCH: nl_context_search}

    def get_nl_context(self) -> str:
        result = {
            PLANNER: {USER: self.nl_context[PLANNER], SYSTEM: self.system_context_planner},
            SEARCH: {USER: self.nl_context[SEARCH], SYSTEM: self.system_context_search}
        }
        return result
    
class SayCanContextModule(ContextModule):
    def __init__(self, config, num_envs):
        self.config = config
        self.num_envs = num_envs
        self.nl_context = {}
        self.simplify_keys = self.config.get(SIMPLIFY_KEYS, False) # by default, don't simplify keys
        self.system_context_planner = self.config[SAYCAN][SYSTEM_PROMPT]
        self.context_obs_dict = {}
        self.house_graph = HouseGraph(self.config[GRAPH])
        self.current_mapping = {}
        self.correct_mapping = {}
        
    def reset(self, key_translator=None, house_logger=None):
        super().reset(key_translator, house_logger)

    def update(self, context_obs_dict):
        """this is called every single step"""
        self.context_obs_dict = context_obs_dict
        self.update_graph()
        nl_context_planner = self._format_prompt(
            self.config[SAYCAN][BASE_PROMPT], context_obs_dict, which_model=SAYCAN
        )
        self.nl_context = {PLANNER: nl_context_planner}

    def get_nl_context(self) -> str:
        result = {
            PLANNER: {USER: self.nl_context[PLANNER], SYSTEM: self.system_context_planner},
        }
        return result
    
    def get_debug_info(self) -> dict:
        # overriding parent function to include locally observed objects and recs
        correct_mapping = self.context_obs_dict[CORRECT_MAPPING]
        current_mapping = self.context_obs_dict[CURRENT_MAPPING]
        # additionally get the locally observed objects and recs
        rooms = self.context_obs_dict[ROOM]
        if self.key_translator.simplify_keys:
            translated_correct_mapping = {}
            for k in correct_mapping:
                vals = correct_mapping[k]
                translated_vals = [self.key_translator.translate(v, simplify=True)[VAL] for v in vals]
                translated_correct_mapping[self.key_translator.translate(k, simplify=True)[VAL]] = translated_vals
            correct_mapping = translated_correct_mapping
            current_mapping = {self.key_translator.translate(k, simplify=True)[VAL]:self.key_translator.translate(current_mapping[k], simplify=True)[VAL] for k in current_mapping}
        return {CORRECT_MAPPING: correct_mapping, CURRENT_MAPPING: current_mapping, ROOM: rooms, GRAPH: self.house_graph, ALL: self.context_obs_dict}

# This is a graph which summarises the objects and recs observed by the agent
class HouseGraph:
    def __init__(self, config):
        self.config = config
        self.mode = self.config.get(MODE, "raw") # raw form or nl
        self.graph = OrderedDict()
        self.gripped_obj = NOTHING
        self.nl_graph = ''
        self.observed_obj_rec = {}
        # simplified translated objects and recs names, not included in diff calculation
        self.local_observation_objs = []
        self.local_observation_recs = []
        
    def reset(self):
        self.graph = OrderedDict()
        self.gripped_obj = NOTHING
        self.nl_graph = ''
        self.observed_obj_rec = {}
        self.local_observation_objs = []
        self.local_observation_recs = []
        
    def deep_copy(self, graph):
        new_graph = OrderedDict()
        for room in graph:
            new_graph[room] = {REC: graph[room][REC][:], OBJ:{k: graph[room][OBJ][k] for k in graph[room][OBJ]}}
        return new_graph, self.gripped_obj
        
    def snapshot(self):
        # snapshot the graph at the current time, for future comparison
        self.graph_snapshot, self.gripped_obj_snapshot = self.deep_copy(self.graph)
        
    def get_observed(self):
        return self.observed_obj_rec

    def get_local_observations(self):
        print ('local obs object', self.local_observation_objs, 'local obs recs', self.local_observation_recs)
        return {OBJ: self.local_observation_objs, REC: self.local_observation_recs}
    
    def get_all_discovered_objects(self):
        # get a list of all found objects (simplified translated names)
        obj_list = []
        for room in self.graph:
            objs = list(self.graph[room][OBJ].keys())
            obj_list.extend(objs)
        return obj_list
        
    def diff(self):
        # calculate the diff between the current graph and the snapshot
        discovered_recs = set()
        discovered_obj = set()
        moved_obj = {}
        new_obj = {}
        old_obj = {}
        if self.gripped_obj_snapshot != NOTHING:
            old_obj[self.gripped_obj_snapshot] = AGENT
        if self.gripped_obj != NOTHING:
            new_obj[self.gripped_obj] = AGENT
        # first check the gripped objects
        for room in self.graph_snapshot:
            new_rec = set(self.graph[room][REC])
            old_rec = set(self.graph_snapshot[room][REC])
            for obj in self.graph_snapshot[room][OBJ]:
                old_obj[obj] = self.graph_snapshot[room][OBJ][obj][REC]
            for obj in self.graph[room][OBJ]:
                new_obj[obj] = self.graph[room][OBJ][obj][REC]
            discovered_recs.update(new_rec - old_rec)
        
        for obj in new_obj:
            if obj not in old_obj:
                discovered_obj.add(obj)
            elif old_obj[obj] != new_obj[obj]:
                moved_obj[obj] = (old_obj[obj], new_obj[obj])
        return list(discovered_obj), list(discovered_recs), moved_obj
        
    def update(self, rooms: list, visible_objects: dict, visible_recs: dict, gripped_obj: str):
        # Input:
        # visible_objects: {'laptop 1':{"room": "kitchen 1", "rec": "kitchen 1 sink 2"}, ...}
        # visible_recs: {"kitchen 1 sink 2": "kitchen 1", ...}
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
        self.gripped_obj = gripped_obj
        if len(self.graph) == 0:
            # initialisation
            self.graph = OrderedDict()
            for room in rooms:
                self.graph[room] = {REC: [], OBJ: {}}
        for rec in visible_recs:
            room = visible_recs[rec]
            if room in self.graph and not rec in self.graph[room][REC]:
                self.graph[room][REC].append(rec)
            self.observed_obj_rec[rec] = True
        for obj in visible_objects:
            self._search_and_pop_object(obj)
            room = visible_objects[obj][ROOM]
            rec = visible_objects[obj][REC]
            if room is None and rec.lower() == 'agent':
                # the agent gripping the object
                continue
            else:
                # add in the object
                self.graph[room][OBJ][obj] = {REC: rec}
                self.observed_obj_rec[rec] = True
            self.observed_obj_rec[obj] = True
        self.local_observation_objs = list(visible_objects.keys())[:]
        self.local_observation_recs = list(visible_recs.keys())[:]
        self.convert_graph_to_text()
        
    def _search_and_pop_object(self, obj):
        for room in self.graph:
            if obj in self.graph[room][OBJ]:
                self.graph[room][OBJ].pop(obj, None)
                break
                
    def convert_graph_to_text(self):
        # to get a string that summarises the graph, to be used in the prompts
        if self.mode == NL:
            descriptions = []
            for room, data in self.graph.items():
                if len(data[REC]) == 0:
                    room_description = f"\nIn {room}, no receptacles found yet."
                else:
                    room_description = f"\nIn {room}, found receptacles: {', '.join(data[REC])}."
                object_descriptions = []
                for obj, receptacle in data[OBJ].items():
                    object_descriptions.append(f"Object {obj} found on {receptacle[REC]}")
                if object_descriptions:
                    room_description += " " + ". ".join(object_descriptions)
                descriptions.append(room_description)
            self.nl_graph = " ".join(descriptions)
        elif self.mode == RAW:
            self.nl_graph = "Here is a graph of currently found recs and objs in the house: " + json.dumps(dict(self.graph))
        return self.nl_graph
