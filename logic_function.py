

class Perceptron:

    def __init__(self, T):
        self.T = T  # threshold

    def get_sum(self, input_vector, weights):
        """Get combined sum of inputs and weights.
        :param input_vector: should be a vector
        :param weights: should be a list the same size as input vector
        """
        S = sum([input_ * weight for input_, weight in zip(input_vector, weights)])
        return S

    def activation_func(self, S):
        if S >= self.T:
            return 1
        return 0
