import logic_function


# AND logic function
AND_perceptron = logic_function.Perceptron(T=1)
print('\nAND function -----------------------')
sum1_and = AND_perceptron.get_sum(input_vector=[0, 0], weights=[0.5, 0.5])
print(f'Expect 0 for vector [0, 0]. Got {AND_perceptron.activation_func(sum1_and)}')

sum2_and = AND_perceptron.get_sum(input_vector=[0, 1], weights=[0.5, 0.5])
print(f'Expect 0 for vector [0, 1]. Got {AND_perceptron.activation_func(sum2_and)}')

sum3_and = AND_perceptron.get_sum(input_vector=[1, 0], weights=[0.5, 0.5])
print(f'Expect 0 for vector [1, 0]. Got {AND_perceptron.activation_func(sum3_and)}')

sum4_and = AND_perceptron.get_sum(input_vector=[1, 1], weights=[0.5, 0.5])
print(f'Expect 1 for vector [1, 1]. Got {AND_perceptron.activation_func(sum4_and)}')

# OR logic function
OR_perceptron = logic_function.Perceptron(T=0.5)
print('\nOR function -----------------------')
sum1_or = OR_perceptron.get_sum(input_vector=[0, 0], weights=[1, 1])
print(f'Expect 0 for vector [0, 0]. Got {OR_perceptron.activation_func(sum1_or)}')

sum2_or = OR_perceptron.get_sum(input_vector=[0, 1], weights=[1, 1])
print(f'Expect 1 for vector [0, 1]. Got {OR_perceptron.activation_func(sum2_or)}')

sum3_or = OR_perceptron.get_sum(input_vector=[1, 0], weights=[1, 1])
print(f'Expect 1 for vector [1, 0]. Got {OR_perceptron.activation_func(sum3_or)}')

sum4_or = OR_perceptron.get_sum(input_vector=[1, 1], weights=[1, 1])
print(f'Expect 1 for vector [1, 1]. Got {OR_perceptron.activation_func(sum3_or)}')

# NOT logic function
NOT_perceptron = logic_function.Perceptron(T=0)
print('\nNOT function -----------------------')
sum1_not = NOT_perceptron.get_sum(input_vector=[0], weights=[-1])
print(f'Expect 1 for vector [0]. Got {NOT_perceptron.activation_func(sum1_not)}')

sum2_not = NOT_perceptron.get_sum(input_vector=[1], weights=[-1])
print(f'Expect 0 for vector [1]. Got {NOT_perceptron.activation_func(sum2_not)}')
