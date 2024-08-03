from weakref import WeakValueDictionary
from typing import Union
import numpy as np
from task import GalfitTask, Config
from components import *
from copy import deepcopy


class State:
    def __init__(self, comp1=None, comp2=None, comp3=None, sky=None) -> None:
        self.comp1 = comp1
        self.comp2 = comp2
        self.comp3 = comp3
        self.sky = sky

    def gen_task(self, task: GalfitTask):
        task.components.clear()
        if self.comp1 is not None:
            task.add_component(self.comp1)
        if self.comp2 is not None:
            task.add_component(self.comp2)
        if self.comp3 is not None:
            task.add_component(self.comp3)
        if self.sky is not None:
            task.add_component(self.sky)

    def split_comp(self, fixed_index=None):
        if self.comp2 is None:
            self.comp2 = deepcopy(self.comp1)
            self.comp1.magnitude += 0.75
            self.comp2.magnitude += 0.75
            self.comp2.effective_radius *= 2
            if fixed_index is not None:
                self.comp2.set_sersic_index(fixed_index, False)
            else:
                self.comp2.set_sersic_index(1, True)
        elif self.comp3 is None:
            self.comp3 = deepcopy(self.comp2)
            self.comp2.magnitude += 0.75
            self.comp3.magnitude += 0.75
            self.comp3.effective_radius *= 2
            if fixed_index is not None:
                self.comp3.set_sersic_index(fixed_index, False)
            else:
                self.comp3.set_sersic_index(1, True)


class GalfitEnv:
    def __init__(self, input_file) -> None:
        config = Config(input_file)
        self.task = GalfitTask(config)
        sersic, sky = self.task.init_guess()
        self.current_state = State(comp1=sersic, sky=sky)
        self.current_state.gen_task(self.task)
        self.task.run()
        self.task.read_component('output/galfit.01')

    def do_action(self, action: int):
        end_flag = True
        if action < 4 and self.current_state[0] != action:
            end_flag = False
        elif action < 8 and self.current_state[1] != action - 4:
            end_flag = False
        elif action < 12 and self.current_state[2] != action - 8:
            end_flag = False
        elif action < 14 and self.current_state[3] != action - 12:
            end_flag = False
        if not end_flag:
            self.task.components.clear()
