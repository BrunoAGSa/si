class OneHotEncoder:

    # init with padder and max lenght

    def __init__(self, padder = None, max_length = None):
        self.padder = padder
        self.max_length = max_length

        self.alphabet = []
        self.char_to_index = {}
        self.index_to_char = {}

    def fit(self, data: list[str]):
        
        # set alphabet
        self.alphabet = list(set(''.join(data)))

        # set char_to_index
        for i, char in enumerate(self.alphabet):
            self.char_to_index[char] = i
        
        # set index_to_char
        for i, char in enumerate(self.alphabet):
            self.index_to_char[i] = char

        if self.max_length is None:
            self.max_length = max([len(word) for word in data])


    def transform(self, data: list[str]):

        # trim the sequences to max_length
        if self.max_length is not None:
            data = [word[:self.max_length] for word in data]

        # pad the sequences with padding character
        if self.padder is not None:
            data = [word + self.padder * (self.max_length - len(word)) for word in data]
       
        # encode the data to the one hot encoded matrices (zeros with a 1 in each line(position) corresponding to the respective character (alphabet)), i.e., for each sequence you wil end up with a matrix of shape max_length x alphabet_size
        encoded_data = []
        for word in data:
            encoded_word = []
            for char in word:
                encoded_char = [0] * len(self.alphabet)
                encoded_char[self.char_to_index[char]] = 1
                encoded_word.append(encoded_char)
            encoded_data.append(encoded_word)

        return encoded_data
    
    def fit_transform(self, data: list[str]):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        # Convert the one-hot encoded matrices back to sequences using the index_to_char dictionary
        decoded_data = []
        for word in data:
            decoded_word = []
            for char in word:
                decoded_word.append(self.index_to_char[char.index(1)])
            decoded_data.append(decoded_word)
        return decoded_data

if __name__ == "__main__":

    data = ['b', 'r', 'u', 'n', 'o', 's', 'a']
    data_words = ["as", "df", "gh", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl", "jk", "la", "sd", "fg", "hj", "kl"]

    encoder = OneHotEncoder(padder=None, max_length=None)
    encoder.fit(data)
    print(encoder.alphabet)
    print(encoder.transform(data))
    print(encoder.inverse_transform(encoder.transform(data)))
    print()


    # using sklearn 
    print()
    print("Sklearn")
    print()

    from sklearn.preprocessing import OneHotEncoder as OneHotEncoder_sklearn
    import numpy as np

    data_skl = np.array([['b'], ['r'], ['u'], ['n'], ['o'], ['s'], ['a']])
    encoder_skl = OneHotEncoder_sklearn(sparse_output=False)
    encoder_skl.fit(data_skl)
    print(encoder_skl.transform(data_skl))
    print(encoder_skl.inverse_transform(encoder_skl.transform(data_skl)))
