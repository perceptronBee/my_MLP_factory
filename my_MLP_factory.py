import math
import random

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def MSE_func(expected_output,output):
    MSE = 0
    for i in range(len(output)):
        MSE+=(expected_output[i]-output[i])**2/2
    return MSE

class perceptron:
    def __init__(self,input_count,nth_neuron):
        self.error= 0
        self.input_count=input_count
        self.nth_neuron=nth_neuron
        self.output= None
        self.bias= random.uniform(-0.5,0.5)
        self.input = None
        self.nonactivated_output= None
        self.w_list = [random.uniform(-0.5,0.5) for _ in range(input_count)]

    def predict(self,input : list):
        self.input=input
        summation=0
        for i in range(self.input_count):
            summation += self.w_list[i]*input[i]
        summation+=self.bias
        self.nonactivated_output=summation
        self.output=sigmoid(summation)

class MLP:
    def __init__(self,neuron_structure : list,input_count_for_each_input_neuron):
        self.neuron_structure=neuron_structure
        self.neuron_list=list()
        self.expected_output = list()
        for nth_layer in range(len(neuron_structure)):
            self.neuron_list.append(list())
            for nth_neuron in range(neuron_structure[nth_layer]):
                if nth_layer == 0:
                    self.neuron_list[nth_layer].append(perceptron(input_count_for_each_input_neuron,nth_neuron))
                else:
                    self.neuron_list[nth_layer].append(perceptron(neuron_structure[nth_layer-1], nth_neuron))

    def feedforward(self,ff_input : list,expected_output):
        self.expected_output=expected_output
        for layer in self.neuron_list:
            for neuron in layer:
                neuron.predict(ff_input)
            ff_input = [neuron.output for neuron in layer]
        return ff_input

    def backpropagation(self,learning_rate):
        for neuron_i,neuron in enumerate(self.neuron_list[-1]):
            neuron.error = (self.expected_output[neuron_i]-neuron.output)*sigmoid_derivative(neuron.output)

        for layer_i in range(len(self.neuron_list)-2,-1,-1):
            for neuron_i,neuron in enumerate(self.neuron_list[layer_i]):
                neuron.error = sum(next_neuron.error * next_neuron.w_list[neuron_i]
                                   for next_neuron in self.neuron_list[layer_i+1])
                neuron.error *= sigmoid_derivative(neuron.output)

        for layer in self.neuron_list:
            for neuron in layer:
                for w_index in range(len(neuron.w_list)):
                    neuron.w_list[w_index] += learning_rate * neuron.error * neuron.input[w_index]
                neuron.bias += learning_rate * neuron.error

    def train_epoch(self, data, labels, learning_rate):
        total_loss = 0
        for x, y in zip(data, labels):
            output = self.feedforward(x, y)
            self.backpropagation(learning_rate)
            total_loss += MSE_func(y, output)
        return total_loss / len(data)

# XOR problemi
data = [[0,0],[0,1],[1,0],[1,1]]
labels = [[0],[1],[1],[0]]

mlp = MLP([2,2,1],2)
epochs = 100000
lr = 0.5

for epoch in range(epochs):
    loss = mlp.train_epoch(data, labels, lr)
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print("\nXOR Test Sonuçları:")
for x, y in zip(data, labels):
    out = mlp.feedforward(x, y)
    print(f"{x} -> {out}")
