

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


def predict_for_and(vector, expected_result):
    """ Моделювання логічної функції І (AND) """
    AND_perceptron = Perceptron(T=1)
    sum_and = AND_perceptron.get_sum(input_vector=vector, weights=[0.5, 0.5])
    print(f'Expect {expected_result} for vector {vector}. Got {AND_perceptron.activation_func(sum_and)}')


def predict_for_or(vector, expected_result):
    """ Моделювання логічної функції АБО (OR) """
    OR_perceptron = Perceptron(T=0.5)
    sum_and = OR_perceptron.get_sum(input_vector=vector, weights=[1, 1])
    print(f'Expect {expected_result} for vector {vector}. Got {OR_perceptron.activation_func(sum_and)}')


def predict_for_not(vector, expected_result):
    """ Моделювання логічної функції НІ (NOT) """
    NOT_perceptron = Perceptron(T=0)
    sum1_not = NOT_perceptron.get_sum(input_vector=vector, weights=[-1])
    print(f'Expect {expected_result} for vector {vector}. Got {NOT_perceptron.activation_func(sum1_not)}')


def predict_for_xor(vector, expected_result):
    """ Моделювання логічної функції виключне АБО (XOR) """
    XOR_perceptron = Perceptron(T=1)
    sum1 = XOR_perceptron.get_sum(input_vector=vector, weights=[1, -1])
    y1 = XOR_perceptron.activation_func(sum1)
    sum2 = XOR_perceptron.get_sum(input_vector=vector, weights=[-1, 1])
    y2 = XOR_perceptron.activation_func(sum2)
    sum_xor = XOR_perceptron.get_sum(input_vector=[y1, y2], weights=[1, 1])
    print(f'Expect {expected_result} for vector {vector}. Got {XOR_perceptron.activation_func(sum_xor)}')


def predict_additional(vector, expected_result):
    """Додаткове завдання: модель для таблиці істинності
    x1 | x2 | x3 | y
    0  | 0  | 0  | 1
    0  | 1  | 0  | 1
    1  | 0  | 0  | 0
    1  | 1  | 1  | 1
    """
    perceptron_ = Perceptron(T=0)
    sum_ = perceptron_.get_sum(input_vector=vector, weights=[-1, 1, 1])
    print(f'Expect {expected_result} for vector {vector}. Got {perceptron_.activation_func(sum_)}')
