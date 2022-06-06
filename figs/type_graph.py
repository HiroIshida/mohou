from graphviz import Digraph

from mohou.types import ElementBase
from mohou.utils import get_all_concrete_leaftypes

if __name__ == "__main__":
    dg = Digraph(format="png")
    visited_set = set()

    def back_recursion(t, is_concrete=False):
        if t in visited_set:
            return
        visited_set.add(t)

        dg.node(t.__name__, style="filled" if is_concrete else None, fillcolor="lightgrey")

        for t_parent in t.__bases__:
            back_recursion(t_parent)
            dg.edge(t_parent.__name__, t.__name__)

    for t in get_all_concrete_leaftypes(ElementBase):
        back_recursion(t, is_concrete=True)
    dg.render("graph", cleanup=True)
