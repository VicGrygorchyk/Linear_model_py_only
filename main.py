import logic_function


def predict_for_and(vector, expected_result):
    """ Моделювання логічної функції І (AND) """
    AND_perceptron = logic_function.Perceptron(T=1)
    sum_and = AND_perceptron.get_sum(input_vector=vector, weights=[0.5, 0.5])
    print(f'Expect {expected_result} for vector {vector}. Got {AND_perceptron.activation_func(sum_and)}')


def predict_for_or(vector, expected_result):
    """ Моделювання логічної функції АБО (OR) """
    OR_perceptron = logic_function.Perceptron(T=0.5)
    sum_and = OR_perceptron.get_sum(input_vector=vector, weights=[1, 1])
    print(f'Expect {expected_result} for vector {vector}. Got {OR_perceptron.activation_func(sum_and)}')


def predict_for_not(vector, expected_result):
    """ Моделювання логічної функції НІ (NOT) """
    NOT_perceptron = logic_function.Perceptron(T=0)
    sum1_not = NOT_perceptron.get_sum(input_vector=vector, weights=[-1])
    print(f'Expect {expected_result} for vector {vector}. Got {NOT_perceptron.activation_func(sum1_not)}')


def predict_for_xor(vector, expected_result):
    """ Моделювання логічної функції виключне АБО (XOR) """
    XOR_perceptron = logic_function.Perceptron(T=1)
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
    perceptron_ = logic_function.Perceptron(T=0)
    sum_ = perceptron_.get_sum(input_vector=vector, weights=[-1, 1, 1])
    print(f'Expect {expected_result} for vector {vector}. Got {perceptron_.activation_func(sum_)}')


# Single Layer Perceptron
print('\nAND function -----------------------')
predict_for_and([0, 0], 0)
predict_for_and([0, 1], 0)
predict_for_and([1, 0], 0)
predict_for_and([1, 1], 1)
print('\nOR function -----------------------')
predict_for_or([0, 0], 0)
predict_for_or([0, 1], 1)
predict_for_or([1, 0], 1)
predict_for_or([1, 1], 1)
print('\nNOT function -----------------------')
predict_for_not([0], 1)
predict_for_not([1], 0)
# XOR function for multilayer perceptron
print('\nXOR function -----------------------')
predict_for_xor([1, 1], 0)
predict_for_xor([1, 0], 1)
predict_for_xor([0, 1], 1)
predict_for_xor([0, 0], 0)
print('\nAdditional model -------------------')
predict_additional([0, 0, 0], 1)
predict_additional([0, 1, 0], 1)
predict_additional([1, 0, 0], 0)
predict_additional([1, 1, 1], 1)
