from final_ACE_framework.graph_util import plot_dot_graph
from lecture6.layer import Layer


class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)
