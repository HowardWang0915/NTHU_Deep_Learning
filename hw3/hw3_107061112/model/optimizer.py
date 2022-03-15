import numpy as np

class SGD():
    def __init__(self, lr, decay=1e-6):
        self.params = None
        self.lr = self.init_lr = lr
        self.iterations = 0
        self.decay = decay

    def step(self, model):
        # Initialize the model if not present
        if self.params == None:
            self.params = self.__copy_weights(model.weights)

        # Learning rate decay
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        for layer in model.layers:
            for key in layer.params.keys():
                self.params[key] = self.lr * layer.grads['d' + key]
                layer.params[key] -= self.params[key]
        self.iterations += 1

    def __copy_weights(self, params):
        result = {}
        for key in params.keys():
            result[key] = np.zeros_like(params[key])
        return result
