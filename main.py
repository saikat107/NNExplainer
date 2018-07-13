import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim import *
from sklearn.metrics import  accuracy_score

import util


def generate_index_array(string):
    int_arr = []
    for char in string:
        int_arr.append(ord(char))
    return int_arr


def generate_one_hot_vector(num_class, idx):
    return np.eye(num_class, dtype='int64')[idx]


def generate_sequence_of_one_hot_vectors(string):
    idx_arr = generate_index_array(string)
    one_hot_seq = [generate_one_hot_vector(idx) for idx in idx_arr]
    return np.asarray(one_hot_seq)


def get_activation_status(values):
    return [1 if value > 0 else 0 for value in values]


def calculate_test_accuracy(model, x, y):
    test_ident_var_arr = [string_to_index_variable(string) for string in x]
    predicted_classes = []
    for i, a in enumerate(test_ident_var_arr):
        output, activation_status, hidden = predict(model, a)
        top_n, top_i = output.data.topk(2)
        predicted_classes.append(top_i[0][0])
    return accuracy_score(y, predicted_classes)


class Model(nn.Module):
    def __init__(self, input_vocab_size, emb_size, encoder_hidden_dim, hidden_sizes, output_size, ):
        super(Model, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_dim = output_size
        self.hidden_dims = hidden_sizes
        self.hidden_dim = encoder_hidden_dim
        self.emb_dim = emb_size
        self.dropout = nn.Dropout(p=0.3)

        self.emb = nn.Embedding(self.input_vocab_size, self.emb_dim)
        self.i2h = nn.Linear(self.emb_dim + self.hidden_dim, self.hidden_dim)

        self.layers = [nn.Linear(self.hidden_dim, self.hidden_dims[0])]
        for i in range(0, len(self.hidden_dims) -1):
            inp_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i+1]
            self.layers.append(nn.Linear(inp_dim, out_dim))

        self.h2o = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor, masks=None):
        # if masks is not None:
        #     assert len(masks) == len(self.layers)
        # else:
        #     masks = []
        #     for layer in self.layers:
        #         mask = Variable(torch.FloatTensor(np.ones(layer.out_features)))
        #         masks.append(mask)
        # util.debug(len(input_tensor))
        hidden = self.initHidden()
        # util.debug(hidden.shape)
        for i in input_tensor[0]:
            hidden = self.iteration(i, hidden)
            # util.debug(hidden.shape)
        output = hidden
        output = self.dropout(output)
        # util.debug(output.shape)
        activation_statuses = []
        for i, layer in enumerate(self.layers):
            output = layer(output)
            # output = output * masks[i]
            output = self.activation(output)
            # util.debug(output.shape)
            activation_status = get_activation_status(list(output[0].data))
            activation_statuses.append(activation_status)
            # util.debug(len(activation_statuses))
            # output = self.dropout(output)
        output = self.h2o(output)
        output = self.softmax(output)
        # util.debug(hidden.data.shape)
        return output, activation_statuses, list(hidden.data)

    def iteration(self, input, hidden):
        embed = self.emb(input)
        embed = torch.unsqueeze(embed, dim=0)
        combined = torch.cat((embed, hidden), -1)
        hidden = self.i2h(combined)
        return  hidden

    def update_parameters(self, learning_rate):
        for p in self.parameters():
            p.data.add_(-learning_rate, p.grad.data)

    def initHidden(self):
        return Variable(torch.FloatTensor(np.zeros((1, self.hidden_dim))))


def train(model, criterion, optimizer, string_tensor, class_tensor):
    model.zero_grad()
    output, _, _ = model(string_tensor)
    loss = criterion(output, class_tensor.view(-1))
    loss.backward()
    optimizer.step()
    return loss.data


def batch_train(model, criterion, optimizer, string_tensor_batch, class_tensor_batch):
    model.zero_grad()
    loss = 0
    for a, b in zip(string_tensor_batch, class_tensor_batch):
        output, _, _ = model(a)
        loss += criterion(output, b)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def predict(model, string_tensor):
    output, activation_status, hidden = model(string_tensor)
    return output, activation_status, hidden


def string_to_index_variable(string):
    idx_arr = [generate_index_array(string)]
    return Variable(torch.LongTensor(idx_arr))


def make_batches(X, Y, batch_size):
    X = np.asarray(X)
    Y = np.asarray(Y)
    num_examples = X.shape[0]
    if batch_size > num_examples:
        raise ValueError
    X_batches = []
    Y_batches = []
    num_batches = int(num_examples / batch_size)
    for i in range(num_batches):
        indices = range(i*batch_size, (i+1)* batch_size)
        X_batches.append(list(X[list(indices)]))
        Y_batches.append(list(Y[list(indices)]))
    return X_batches, Y_batches


def load_data(file_path):
    with open(file_path) as fp:
        train_x = []
        train_y = []
        for line in fp:
            parts = line.split('\t')
            train_x.append(parts[0])
            train_y.append(int(parts[1]))
        fp.close()
        return train_x, train_y


def main():
    positive_class_probs = []
    negative_class_probs = []
    criterion = nn.CrossEntropyLoss()
    n_epochs = 10
    train_identifiers, train_classes = load_data('data/train_tokens.csv')
    valid_identifiers, valid_classes = load_data('data/valid_tokens.csv')
    valid_ident_var_arr = [string_to_index_variable(string) for string in valid_identifiers]
    valid_classes_var_arr = [Variable(torch.LongTensor([class_])) for class_ in valid_classes]
    buggy_example_identifiers = None
    buggy_example_classes = None
    for iter in range(10):
        if buggy_example_identifiers is not None and buggy_example_classes is not None:
            train_identifiers.extend(buggy_example_identifiers)
            train_classes.extend(buggy_example_classes)
        buggy_example_identifiers = None
        buggy_example_classes = None
        model = Model(input_vocab_size=128, emb_size=32,
                      encoder_hidden_dim=16, hidden_sizes=[8,4], output_size=2)
        optimizer = Adam(model.parameters())#SGD(model.parameters(),lr=lr)
        losses = train_all(model, n_epochs, criterion, train_classes, optimizer, train_identifiers)
        # torch.save(model, 'model.bin')
        # model = torch.load('model.bin')
        predicted_classes = []
        positive_class_activations = []
        negative_class_activations = []
        num_correct = 0
        for i, a in enumerate(valid_ident_var_arr):
            output, activation_status, hidden = predict(model, a)
            b = valid_classes_var_arr[i]
            top_n, top_i = output.data.topk(2)
            all_activations = []
            for x in activation_status:
                all_activations.extend(x)
            actual_class = int(b.data[0])
            predicted_class = int(top_i[0][0])
            if actual_class == 1:
                positive_class_activations.append(all_activations)
            else:
                negative_class_activations.append(all_activations)
            correctly_classified = (predicted_class == actual_class)
            if not correctly_classified:
                if buggy_example_identifiers is None:
                    buggy_example_identifiers = [valid_identifiers[i]]
                    buggy_example_classes = [actual_class]
                else:
                    buggy_example_identifiers.append(valid_identifiers[i])
                    buggy_example_classes.append(actual_class)
            else:
                num_correct += 1
            predicted_classes.append(predicted_class)
        # util.debug(buggy_example_identifiers)
        # util.debug(buggy_example_classes)
        accuracy = num_correct * 1.0 / len(valid_classes_var_arr)
        util.debug('Iteration : ', iter, '\tAccuracy: ', accuracy,
                   '\tMisclassified : ', len(buggy_example_classes), )
        positive_probs, negative_probs = post_process_activation_statuses(
                                positive_class_activations, negative_class_activations)
        positive_class_probs.append(positive_probs)
        negative_class_probs.append(negative_probs)
    write_to_csv(positive_class_probs, 'positive_probabilities.csv')
    write_to_csv(negative_class_probs, 'negative_probabilities.csv')


def write_to_csv(list_of_list, file_name):
    with open(file_name, 'w') as file:
        for list_of_values in list_of_list:
            string = format_double_list(list_of_values)
            file.write(string + '\n')
        file.close()


def train_all(model, n_epochs, criterion, train_classes, optimizer, train_identifiers):
    losses = []
    X_ = [string_to_index_variable(string) for string in train_identifiers]
    Y_ = [Variable(torch.LongTensor([[int(class_)]])) for class_ in train_classes]
    import datetime as dt
    for epoch in range(n_epochs):
        n1 = dt.datetime.now()
        total_loss = 0
        for X, Y in zip(X_, Y_):
            l = train(model, criterion=criterion, optimizer=optimizer,
                      string_tensor=X, class_tensor=Y)
            total_loss += l
        n2 = dt.datetime.now()
        elapsed_time = (n2 - n1).seconds
        # util.debug('Epoch : ', epoch, '\tLoss: ', float(total_loss),
        #            '\tTime Elapsed : ', elapsed_time, ' seconds')
        losses.append(total_loss)
    return losses


def post_process_activation_statuses(positive_class_activations, negative_class_activations):
    positive_class_activations = np.asarray(positive_class_activations)
    negative_class_activations = np.asarray(negative_class_activations)
    positive_class_activation_probability_vector = create_probability_vector(
                                        positive_class_activations)
    negative_class_activation_probability_vector = create_probability_vector(
                                        negative_class_activations)
    return positive_class_activation_probability_vector, \
           negative_class_activation_probability_vector
    pass


def create_probability_vector(class_activations):
    total_instances = class_activations.shape[0]
    activations = np.sum(class_activations, axis=0)
    probs = activations/total_instances
    return probs
    pass


def format_double_list(l):
    string_arr = [str(x) for x in l]
    string = ','.join(string_arr)
    return string


if __name__ == '__main__':
    main()
