import keras        
import sys
import h5py
import os.path
import argparse
import numpy                    as np
from keras.models               import Sequential
from keras.layers               import LSTM, Dense, Dropout
from keras.utils                import np_utils
from keras.callbacks            import ModelCheckpoint
from keras                      import backend as K

SEQ_LENGTH = 100

def load_data(filename):
    data    = open(filename).read()
    data    = data.lower()

    return data

def get_data_stats(data):
    chars       = sorted(list(set(data)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    vocab_size  = len(chars)
    
    return int_to_char, char_to_int, vocab_size

def get_formatted_data(data, vocab_size, char_to_int):

    print(("Total Characters: ", len(data)))
    print(("Total Vocab: ", vocab_size))

    # Using numpy append is slower
    list_X      = []
    list_Y      = []
    for i in range(0, len(data) - SEQ_LENGTH, 1):
        seq_in  = data[i : i + SEQ_LENGTH]
        seq_out = data[i + SEQ_LENGTH]
        list_X.append([char_to_int[char] for char in seq_in])
        list_Y.append(char_to_int[seq_out])
    
    n_patterns  = len(list_X)

    X           = np.reshape(list_X, (n_patterns, SEQ_LENGTH, 1))
    Y           = np_utils.to_categorical(list_Y)

    return X, Y

def create_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    hidden_dim  = int(hidden_dim)
    model       = Sequential()
    flag        = True 

    if n_layers == 1:   
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2])) )
        model.add( Dropout(drop) )

    else:
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2]), return_sequences = True) )
        model.add( Dropout(drop) )
        for i in range(n_layers - 2):
            model.add( LSTM(hidden_dim, return_sequences = True) )
            model.add( Dropout(drop) )
        model.add( LSTM(hidden_dim) )

    model.add( Dense(n_out, activation = activ) )

    return model

def train(model, X, Y, n_epochs, b_size, vocab_size, **kwargs):    
    loss            = kwargs.get('loss', 'categorical_crossentropy')
    opt             = kwargs.get('optimizer', 'adam')
    
    model.compile(loss = loss, optimizer = opt)

    filepath        = "Weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint      = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list  = [checkpoint]
    X               = X / float(vocab_size)
    model.fit(X, Y, epochs = n_epochs, batch_size = b_size, callbacks = callbacks_list)

def generate_text(model, X, filename, ix_to_char, vocab_size):

    model.load_weights(filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    start   = np.random.randint(0, len(X) - 1)
    pattern = np.ravel(X[start]).tolist()

    print ("Seed:")
    print ("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
    output = []
    for i in range(250):
        x           = np.reshape(pattern, (1, len(pattern), 1))
        x           = x / float(vocab_size)
        
        prediction  = model.predict(x, verbose = 0)
        index       = np.argmax(prediction)
        result      = index
        output.append(result)
        pattern.append(index)
        pattern = pattern[1 : len(pattern)]

    print("Predictions")
    print ("\"", ''.join([ix_to_char[value] for value in output]), "\"")
    print ("\nDone.")

def store_array(x, filename):
    hf5 = h5py.File(filename, 'w')
    hf5.create_dataset('dataset_1', data = x)
    hf5.close()

def read_array(filename):
    hf5 = h5py.File(filename, 'r')
    x   = hf5['dataset_1'][:]
    hf5.close()
    return x

def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument('--mode', default = "train")
    parser.add_argument('--weights', default = None)

    args    = parser.parse_args()

    filename                            = 'data/game_of_thrones.txt'
    weights                             = args.weights 
    data                                = load_data(filename)
    ix_to_char, char_to_ix, vocab_size  = get_data_stats(data)

    print("Unique character mappings: \n ", ix_to_char)

    if not os.path.exists("data/input_data.hdf5") and not os.path.exists("data/output_data.hdf5"):
        X, Y    = get_formatted_data(data, vocab_size, char_to_ix)
        store_array(X, "data/input_data.hdf5")
        store_array(Y, "data/output_data.hdf5")
    else:
        print("Getting data from hdf5 files")
        X = read_array("data/input_data.hdf5")
        Y = read_array("data/output_data.hdf5")

    model   = create_model(1, X.shape, 256, Y.shape[1])

    if args.mode == 'train':
        print("Training")
        train(model, X, Y, 100, 128, vocab_size)
    elif args.mode == 'test':
        K.set_learning_phase(0)
        print("Generating text")
        if weights is None:
            raise Exception("No weights provided")
        else:
            generate_text(model, X, weights, ix_to_char, vocab_size)

if __name__ == '__main__':
    main()
