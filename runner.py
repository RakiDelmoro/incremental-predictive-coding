import gzip
import pickle
from IPC_model.model import model
from utils import image_data_batching

def runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    PARAMS_LR = 0.0001
    ACTIVATION_LR = 0.1
    NUM_ITERATIONS = 8

    with gzip.open('./Mnist_dataset/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    ipc_v3_train, ipc_v3_test = model([784, 64, 64, 64, 10], PARAMS_LR, ACTIVATION_LR, NUM_ITERATIONS)

    t = 0 # For AdamW optimizer let's use a global time step
    for i in range(3000):
        training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
        loss = ipc_v3_train(training_loader, t)
        accuracy = ipc_v3_test(test_loader)
        print(f'EPOCH: {i+1} LOSS: {loss} Accuracy: {accuracy}')
        t += 1
        
runner()