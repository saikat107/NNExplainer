from util import generate_feature_vector_from_string
from sklearn import tree
import graphviz

class DataEntry():
    def __init__(self, h, a, actual, predicted, ori):
        self.hiddens = h
        self.activation_status = a
        self.actual_is_valid_identifier = actual
        self.predicted_is_valid_identifiers = predicted
        self.string = ori
        self.feature_vector = self.create_feature_vector()
        self.num_activation_status = len(self.activation_status)

    def create_feature_vector(self):
        return generate_feature_vector_from_string(self.string)

    def __repr__(self):
        return str(self.hiddens) + ' ' + str(self.activation_status) + ' ' + str(self.actual_is_valid_identifier) + ' '\
                + str(self.predicted_is_valid_identifiers) + ' ' + self.string + ' ' + str(self.feature_vector)


def pre_process(hidden):
    parts = hidden.split(',')
    return [float(x.strip().strip('[').strip(']')) for x in parts]


def main():
    data_file_path = 'test_results.txt'
    file = open(data_file_path)
    data = []
    for line in file:
        hiddens = []
        act_status = []
        parts = line.split('\t')
        hidden = parts[0]
        hiddens = pre_process(hidden)
        num_activation_layers = len(parts) - 4 # hidden, original, predicted, string
        for i in range(num_activation_layers):
            acts = parts[1 + i]
            act_status.extend(pre_process(acts))
        predicted = int(parts[-3])
        original = int(parts[-2])
        string = parts[-1]
        entry = DataEntry(hiddens, act_status, original, predicted, string)
        # print(entry)
        data.append(entry)

    features = [entry.feature_vector for entry in data]
    # for entry in data:
    #     if entry.feature_vector[0] == 0 and entry.feature_vector[1] == False:
    #         print(entry.string, entry.actual_is_valid_identifier, entry.predicted_is_valid_identifiers)

    print(features)
    for i in range(data[0].num_activation_status):
        activation_status = [entry.activation_status[i] for entry in data]
        clf = tree.DecisionTreeClassifier()
        clf.fit(features, activation_status)
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['start_char', 'symbol'], filled=True)
        graph = graphviz.Source(dot_data)
        graph.render('short_status/' + str(i))

    actual_classes = [entry.actual_is_valid_identifier for entry in data]
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, actual_classes)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['start_char', 'symbol'], filled=True)
    graph = graphviz.Source(dot_data)
    graph.render('short_status/Label')

    predicted_class = [entry.predicted_is_valid_identifiers for entry in data]
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, predicted_class)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['start_char', 'symbol'], filled=True)
    graph = graphviz.Source(dot_data)
    graph.render('short_status/predicted')





if __name__ == '__main__':
    main()