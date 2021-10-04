import math
from random import randrange


class PredictNumberModel:

    def __init__(self, size):
        self.size = size
        self.weights = [self.init_rand_weight() for _ in range(self.size)]

    def init_rand_weight(self):
        weight = round(randrange(1, 10) * 0.1, 1)
        print(f'Random weight is {weight}')
        return weight

    def calculate_sum(self, input_vector, weights):
        sum_ = 0
        for x, w in zip(input_vector, weights):
            sum_ += x * w
        print(f'S = {sum_}')
        return sum_

    def cost_function(self, predicted, y):
        """ error = (Y - y) ^ 2 """
        result = (predicted - y) ** 2
        print(f"Cost is {result}")
        return result

    def loss_function(self, costs):
        """ Loss is sqr root of mean of all errors. """
        result = math.sqrt(sum(costs) / len(costs))
        print(f"Loss is {result}")
        return result

    def activation_func(self, input_sum):
        y = 1 / (1 + math.exp(-input_sum)) * 10
        return y

    def find_corrections(self, item, learning_rate=0.1):
        """Find corrections for each weight."""
        weight_corrections = []
        predicted = item['predicted']
        true_label = item['true_label']
        sum_of_weights = item['sum_of_weights']
        for inp in item['inputs']:
            # find derivative
            E = (predicted - true_label) * \
                (
                        math.exp(-sum_of_weights) /
                        ((1 + math.exp(-sum_of_weights)) ** 2)
                )\
                * inp
            # print(f'Derivative for {inp} is {E}')
            weight_delta = (-learning_rate) * E
            # print(f'Weight delta is {weight_delta}')
            weight_corrections.append(weight_delta)
        return weight_corrections

    def back_propagation(self, results):
        """Find a mean for weights correction
        and update all weighs."""
        weight_corrections = []
        for item in results:
            weight_corrections.extend(item['weight_corrections'])

        w_mean = 1 / len(weight_corrections) * sum(weight_corrections)
        # init a corrected set of weights
        self.weights = [weight + w_mean for weight in self.weights]
        print(f'new weights {self.weights}. Length is {len(self.weights)}')

    def forward(self, train_data):
        """Move through whole training data and calculate error for each prediction."""
        start = 0
        end = self.size
        results = []
        while True:
            if end > len(train_data) - 1:
                # no data left for prediction, exit
                break
            # get training sample
            print('\n---------------------------------------------------------')
            data = train_data[start: end]
            label = train_data[end]
            # calculate S
            sum_of_weights = self.calculate_sum(data, self.weights)
            y_result = self.activation_func(sum_of_weights)
            print(f' ======= Predicted {y_result} for inputs [{data}], weights [{self.weights}]. True value is {label}')
            run_result = {
                "inputs": data,
                "sum_of_weights": sum_of_weights,
                "true_label": label,
                "predicted": y_result
            }
            weight_corrections = self.find_corrections(run_result)
            run_result['weight_corrections'] = weight_corrections
            results.append(run_result)
            start += 1
            end += 1
            train_data_length = len(train_data)
            if start >= train_data_length:
                break
        return results

    def train(self, train_data, max_epoch=1_000_000):
        iteration = 0
        while True:
            results = self.forward(train_data)
            # TODO update weights
            self.back_propagation(results)
            costs = [self.cost_function(item['predicted'], item['true_label']) for item in results]
            # TODO find loss
            loss = self.loss_function(costs)
            if loss == 0.1:
                break
            if iteration >= max_epoch:
                print('\nExited: reached max of epoch.')
                break
            iteration += 1

    def test(self, dataset):
        success = 0
        results = self.forward(dataset)
        predicted = [item['predicted'] for item in results]
        labels = [item['true_label'] for item in results]
        for prd, label in zip(predicted, labels):
            if abs(prd - label) <= 0.1:
                success += 1
        print(f'Accuracy is {success / len(labels) * 100}')
