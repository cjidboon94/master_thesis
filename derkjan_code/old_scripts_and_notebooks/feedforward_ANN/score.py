import numpy as np

def get_accuracy(network, X, y):
    network.update_batch_size(1)
    correct_classified = 0
    for count in range(X.shape[0]):
        if np.argmax(network.forward(np.array([X[count]])))==y[count]:
            correct_classified += 1

    #print("the accuracy on the dataset is {}".format(correct_classified/X.shape[0]))
    network.update_batch_size(network.batch_size)
    return correct_classified/X.shape[0]

"""
def get_accuracy2(network, X, y):
    batch_size = network.batch_size
    correct_classified = 0
    
    num_whole_batches = int(len(X[0])/batch_size)
    if num_whole_batches < 1:
        print("the size of the validation set is smaller than the batch size")
        print("zero will be returned")
        return 0

    for count in np.arange(0, num_whole_batches):
        out = network.forward(X[count*batch_size:(count+1)*batch_size])
        print(out)
        out_arg_max = np.argmax(out, 1)
        y_arg_max = y
        #y_arg_max = np.argmax(y[count*batch_size:(count+1)*batch_size], 1)
        correct_classified += np.count_nonzero(y_arg_max==out_arg_max)

    return correct_classified / (num_whole_batches*batch_size)
"""
