from graphviz import Digraph
from typing import Union
from weakref import WeakValueDictionary


class State:
    def __init__(self, name: str = None, reward: float = 0) -> None:
        self._name = name
        self._successor = []
        self._r = reward

    @property
    def name(self):
        return self._name

    @property
    def out_degree(self):
        return len(self._successor)

    @property
    def reward(self):
        return self._r

    def add_successor(self, successor: 'State'):
        self._successor.append(successor)

    def plot(self, dot: Digraph):
        dot.node(self.name, f'{self.name}\nr={self.reward}')
        for successor in self._successor:
            successor.plot(dot)
            dot.edge(self.name, successor.name)


class GalfitEnv:
    def __init__(self) -> None:
        self._root = State('root', 0)
        self._states = [self._root]
        self.current_state = self._root

    def _add_state(self, state: State, parents: Union[State, list[State]]):
        if isinstance(parents, list):
            for parent in parents:
                parent.add_successor(state)
        else:
            parents.add_successor(state)
        self._states.append(state)

    def plot(self):
        dot = Digraph(comment='Galfit Environment')
        self._root.plot(dot)
        dot.render('fig/galfit_env', view=True, format='pdf')

    def do_action(self, action: int):
        pass


if __name__ == '__main__':
    env = GalfitEnv()
    env.plot()
