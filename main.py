from logic_function import (
    predict_for_and, predict_for_or, predict_for_not, predict_additional, predict_for_xor
)


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

print("\nPredict time-series rows.")
