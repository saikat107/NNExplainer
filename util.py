class Dictionary:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_to_idx['UNK'] = 0
        self.idx_to_word[0] = 'UNK'
        self.word_to_idx['SOS'] = 1
        self.idx_to_word[1] = 'SOS'
        self.word_to_idx['EOS'] = 2
        self.idx_to_word[2] = 'EOS'
        self.vocab_size = 3

    def add_sentences(self, sentences):
        for sentence in sentences:
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        words = sentence.split(' ')
        for word in words:
            if word not in self.word_to_idx.keys():
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_sequences(self):
        pass


def length(string):
    return len(string)


def start_char(string):
    char = string[0]
    if char.isalpha():
        return 0
    elif char == '_':
        return 1
    elif char.isdigit():
        return 2
    else:
        return 3


def start_with_underscore(string):
    char = string[0]
    if char == '_':
        return 1
    else:
        return 0


def is_symbol_present(string):
    symbols = list('`!@#$%^&*()-+~[{]}\\|:;\'\",<.>/?=')
    for c in string:
        if symbols.__contains__(c):
            return True
    return False


features = ['start_char', 'is_symbol_present']


def generate_feature_vector_from_string(string, list_of_feature_functions=features):
    features = []
    for feature_function in list_of_feature_functions:
        features.append(eval(feature_function)(string))
    return features


if __name__ == '__main__':
    print(generate_feature_vector_from_string('_jdsfh4*&('))