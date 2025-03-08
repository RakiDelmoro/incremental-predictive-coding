import math
import torch
import random
import numpy as np
from torch.nn import init
from features import RED, GREEN, RESET
from torch.nn.init import kaiming_uniform_

def leaky_relu(input_data, return_derivative=False):
    if return_derivative:
        # x = np.maximum(input_data * 0.05, input_data)
        return np.where(input_data > 0, 1, 0.05 * input_data)
    else:
        return np.maximum(input_data * 0.05, input_data)

def softmax(input_data):
    # Subtract max value for numerical stability
    shifted_data = input_data - np.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = np.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    return exp_data / sum_exp_data

def initializer(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)
    return [np.array(weights), np.array(bias)]

def parameters_init(network_architecture: list):
    parameters = []
    for size in range(len(network_architecture)-1):
        input_size = network_architecture[size]
        output_size = network_architecture[size+1]
        connections = initializer(input_size, output_size)
        parameters.append(connections)
    return parameters

def forward_pass(input_data, parameters):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        bias = parameters[each][1]

        pre_activation = np.matmul(activation, weights) + bias

        activation = leaky_relu(pre_activation) if not last_layer else softmax(pre_activation)
        activations.append(activation)

    return activations

def calculate_gradients(network_activations, expected_output, parameters):
    loss = -np.mean(np.sum(expected_output * np.log(network_activations[-1]), axis=-1))
    error = network_activations[-1] - expected_output
    activations_errors = [error]
    for each in range(len(parameters)-1):
        weights = parameters[-(each+1)][0].T
        error = leaky_relu(network_activations[-(each+2)], return_derivative=True) * np.matmul(error, weights)
        activations_errors.insert(0, error)

    return loss, activations_errors

def update_parameters(activations, activations_error, parameters, lr, m, v, t):
    # TODO: Make this function more readable
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.01

    t += 1
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]
        bias = parameters[-(each+1)][1]

        pre_activation = activations[-(each+2)]
        error = activations_error[-(each+1)]

        nudge = np.matmul(pre_activation.T, error)
        grad_weights = nudge / error.shape[0]
        grad_bias = np.sum(error, axis=0) / error.shape[0]

        m_weights, m_bias = m[-(each+1)]
        v_weights, v_bias = v[-(each+1)]

        m_weights = beta1 * m_weights + (1 - beta1) * grad_weights
        v_weights = beta2 * v_weights + (1 - beta2) * (grad_weights ** 2)

        m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
        v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias ** 2)

        m[-(each+1)] = (m_weights, m_bias)
        v[-(each+1)] = (v_weights, v_bias)

        # Bias correction
        m_hat_weights = m_weights / (1 - beta1**t)
        v_hat_weights = v_weights / (1 - beta2**t)
        m_hat_bias = m_bias / (1 - beta1**t)
        v_hat_bias = v_bias / (1 - beta2**t)

        # Update weights with AdamW (include weight decay)
        weights -= lr * (m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)) #+ weight_decay * weights)
        weights -= lr * (weight_decay * weights)
 
        # Update bias with Adam (no weight decay)
        bias -= lr * (m_hat_bias / (np.sqrt(v_hat_bias) + epsilon))

def initialize_moments(size):
    # TODO: make this function more readable
    m = []
    v = []
    for each in range(len(size)-1):
        w = np.zeros(shape=(size[each], size[each+1]))
        b = np.zeros(shape=(size[each+1]))

        m.append([w, b])
        v.append([w, b])

    return m, v

def model(size: list):
    parameters = parameters_init(size)
    moments, velocity = initialize_moments(size)

    def train_runner(dataloader, t):
        losses = []
        for input_image, label in dataloader:
            model_activations = forward_pass(input_image, parameters)
            loss, activations_gradients = calculate_gradients(model_activations, label, parameters)
            update_parameters(model_activations, activations_gradients, parameters, 0.0001, moments, velocity, t)
            losses.append(loss)

        return np.mean(np.array(losses))

    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            model_pred_probabilities = forward_pass(batched_image, parameters)[-1]
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
            for each in range(len(batched_label)//10):
                model_prediction = model_pred_probabilities[each].argmax()
                if model_prediction == batched_label[each].argmax(axis=-1): correctness.append((model_prediction.item(), batched_label[each].argmax(axis=-1).item()))
                else: wrongness.append((model_prediction.item(), batched_label[each].argmax(axis=-1).item()))
            print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy))
        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]
        return np.mean(np.array(accuracy)).item()

    return train_runner, test_runner
