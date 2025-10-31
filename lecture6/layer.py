from lecture6.parameter import Parameter


class Layer:
    def __init__(self):
        self._params_name = set()  # set 是集合，无序且元素唯一

    def __setattr__(self, name, value):
        # 只搜集Parameter，不搜集Variable
        if isinstance(value, (Parameter, Layer)):
            self._params_name.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # tuple 不可变，转换成 list 类型
        self.inputs, self.outputs = list(inputs), list(outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params_name:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    # 清除所有参数的梯度
    def clear_grads(self):
        for param in self.params():
            param.clear_grad()
