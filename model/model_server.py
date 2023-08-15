import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

class AvaxNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout, loss_fn=nn.CrossEntropyLoss(), optimizer=optim.AdamW, lr=0.001):
        super(AvaxNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])

        # Define the architecture
        self.layers = nn.ModuleList()
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(dropout)

        # Set the loss function and optimizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.parameters(), lr=lr)

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.layers:
            x = torch.tanh(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return nn.Softmax(dim=1)(x)

    def fit(self, train_loader, valid_loader, epochs=1000):
        for epoch in range(epochs):
            self.train()

            LOSS_train = 0.0
            PRECISION_0_train, RECALL_0_train, F1_0_train, ACCURACY_0_train = 0.0, 0.0, 0.0, 0.0
            PRECISION_1_train, RECALL_1_train, F1_1_train, ACCURACY_1_train = 0.0, 0.0, 0.0, 0.0

            for j, data in enumerate(train_loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward (aka predict)
                outputs = self.forward(inputs)
                predictions = np.array(
                    [[1.0, 0.0] if np.argmax(p.cpu().detach().numpy()) == 0 else [0.0, 1.0] for p in outputs])

                # Loss + Backprop
                loss = self.loss_fn(outputs, labels)
                LOSS_train += loss.item()
                loss.backward()
                self.optimizer.step()

                # Metrics
                (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=0)
                PRECISION_0_train += precision
                RECALL_0_train += recall
                F1_0_train += f1
                ACCURACY_0_train += accuracy

                (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=1)
                PRECISION_1_train += precision
                RECALL_1_train += recall
                F1_1_train += f1
                ACCURACY_1_train += accuracy

            # Average the metrics based on the number of batches
            LOSS_train /= len(train_loader)
            PRECISION_0_train /= len(train_loader)
            RECALL_0_train /= len(train_loader)
            F1_0_train /= len(train_loader)
            ACCURACY_0_train /= len(train_loader)
            PRECISION_1_train /= len(train_loader)
            RECALL_1_train /= len(train_loader)
            F1_1_train /= len(train_loader)
            ACCURACY_1_train /= len(train_loader)

            # Validation set
            self.eval()
            with torch.no_grad():

                LOSS_valid = 0.0
                PRECISION_0_valid, RECALL_0_valid, F1_0_valid, ACCURACY_0_valid = 0.0, 0.0, 0.0, 0.0
                PRECISION_1_valid, RECALL_1_valid, F1_1_valid, ACCURACY_1_valid = 0.0, 0.0, 0.0, 0.0

                for k, data in enumerate(valid_loader, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Forward (aka predict)
                    outputs = self.forward(inputs)
                    predictions = np.array(
                        [[1.0, 0.0] if np.argmax(p.cpu().detach().numpy()) == 0 else [0.0, 1.0] for p in outputs])

                    # Loss
                    loss = self.loss_fn(outputs, labels)
                    LOSS_valid += loss.item()

                    # Metrics
                    (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=0)
                    PRECISION_0_valid += precision
                    RECALL_0_valid += recall
                    F1_0_valid += f1
                    ACCURACY_0_valid += accuracy

                    (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=1)
                    PRECISION_1_valid += precision
                    RECALL_1_valid += recall
                    F1_1_valid += f1
                    ACCURACY_1_valid += accuracy

                # Average the metrics based on the number of batches
                LOSS_valid /= len(valid_loader)
                PRECISION_0_valid /= len(valid_loader)
                RECALL_0_valid /= len(valid_loader)
                F1_0_valid /= len(valid_loader)
                ACCURACY_0_valid /= len(valid_loader)
                PRECISION_1_valid /= len(valid_loader)
                RECALL_1_valid /= len(valid_loader)
                F1_1_valid /= len(valid_loader)
                ACCURACY_1_valid /= len(valid_loader)

            ##################
            ### STATISTICS ###
            ##################

            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1}/{epochs}' + '\n')
                print(f'Train Loss: {LOSS_train:.5f}  |  Valid Loss: {LOSS_valid:.5f}')
                print(f'Train Accu: {ACCURACY_0_train:.5f}  |  Valid Accu: {ACCURACY_0_valid:.5f}')
                print()
                print("Class 0:")
                print(f'Train Prec: {PRECISION_0_train:.5f}  |  Valid Prec: {PRECISION_0_valid:.5f}')
                print(f'Train Rcll: {RECALL_0_train:.5f}  |  Valid Rcll: {RECALL_0_valid:.5f}')
                print(f'Train  F1 : {F1_0_train:.5f}  |  Valid  F1 : {F1_0_valid:.5f}')
                print()
                print("Class 1:")
                print(f'Train Prec: {PRECISION_1_train:.5f}  |  Valid Prec: {PRECISION_1_valid:.5f}')
                print(f'Train Rcll: {RECALL_1_train:.5f}  |  Valid Rcll: {RECALL_1_valid:.5f}')
                print(f'Train  F1 : {F1_1_train:.5f}  |  Valid  F1 : {F1_1_valid:.5f}')
                print()
                print('-' * 43)
                print()

        print('Finished Training')

    def get_metrics(self, preds, labels, for_class=1):
        TP, FP, TN, FN = 0, 0, 0, 0

        # Iterate over all predictions
        for idx in range(len(preds)):
            # If we predicted the sample to be class {for_class}
            if preds[idx][for_class] == 1:
                # Then check whether the prediction was right or wrong
                if labels[idx][for_class] == 1:
                    TP += 1
                else:
                    FP += 1
            # Else we predicted another class
            else:
                # Check whether the "not class {for_class}" prediction was right or wrong
                if labels[idx][for_class] != 1:
                    TN += 1
                else:
                    FN += 1

        precision = TP / (TP + FP) if TP + FP > 0 else 0  # Of all "class X" calls I made, how many were right?
        recall = TP / (
                    TP + FN) if TP + FN > 0 else 0  # Of all "class X" calls I should have made, how many did I actually make?
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (TP + TN) / (TP + FP + TN + FN)

        return (precision, recall, f1, accuracy)

    def predict(self, X_input):
        self.eval()
        result = []

        # Predict from PyTorch dataloader
        if type(X_input) == torch.utils.data.dataloader.DataLoader:

            with torch.no_grad():
                for k, data in enumerate(X_input, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Forward (aka predict)
                    outputs = self.forward(inputs)
                    predictions = np.array(
                        [[1.0, 0.0] if np.argmax(p.cpu().detach().numpy()) == 0 else [0.0, 1.0] for p in outputs])

                    if len(result) == 0:
                        result = predictions
                    else:
                        result = np.concatenate((result, predictions))

                return np.array([np.argmax(x) for x in result])

        # Predict from Numpy array or list
        else:

            if type(X_input) == list:
                X_input = np.array(X_input)

            if isinstance(X_input, np.ndarray):
                X_test = torch.from_numpy(X_input).float()

                with torch.no_grad():
                    for k, data in enumerate(X_test):
                        # Get the inputs; data is a list of [inputs]
                        inputs = data.reshape(1, -1).to(self.device)
                        # Forward (aka predict)
                        outputs = self.forward(inputs)
                        predictions = np.array(
                            [[1.0, 0.0] if np.argmax(p.cpu().detach().numpy()) == 0 else [0.0, 1.0] for p in outputs])

                        if len(result) == 0:
                            result = predictions
                        else:
                            result = np.concatenate((result, predictions))

                    return np.array([np.argmax(x) for x in result])

            else:
                raise ("Input must be a dataloader, numpy array or list")

    def predict_proba(self, X_input):
        self.eval()
        result = []

        # Predict from PyTorch dataloader
        if type(X_input) == torch.utils.data.dataloader.DataLoader:

            with torch.no_grad():
                for k, data in enumerate(X_input, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Forward (aka predict)
                    outputs = self.forward(inputs)
                    class_proba = outputs.cpu().detach().numpy()

                    if len(result) == 0:
                        result = class_proba
                    else:
                        result = np.concatenate((result, class_proba))

                return result

        # Predict from Numpy array or list
        else:

            if type(X_input) == list:
                X_input = np.array(X_input)

            if isinstance(X_input, np.ndarray):
                X_input = torch.from_numpy(X_input).float()

                with torch.no_grad():
                    for k, data in enumerate(X_input):
                        # Get the inputs; data is a list of [inputs]
                        inputs = data.reshape(1, -1).to(self.device)
                        # Forward (aka predict)
                        outputs = self.forward(inputs)
                        class_proba = outputs.cpu().detach().numpy()

                        if len(result) == 0:
                            result = class_proba
                        else:
                            result = np.concatenate((result, class_proba))

                    return result

            else:
                raise ("Input must be a dataloader, numpy array or list")


def load_and_process_data(file_path):
    # Load the dataset
    dataset = torch.load(file_path)

    # Split dataset into features and targets
    features = dataset.tensors[0]
    targets = dataset.tensors[1]
    eye_matrix = torch.eye(2)  # makes the possible one-hot labels [1, 0] and [0,1]
    targets = eye_matrix[targets.long()]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=256)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=256)

    return train_loader, test_loader


def main():
    # Load and process the data
    train_loader, test_loader = load_and_process_data('/nas/home/jfu/data/hate_model_dataset.pt')

    # Define the model
    net = AvaxNN(input_size=768, output_size=2, hidden_layers=[512, 256, 128, 64, 32, 16], dropout=0.5)

    # Train the model
    net.fit(train_loader, test_loader, epochs=10)

    # Get the model's predicted probabilities for the test set
    test_probabilities = net.predict_proba(test_loader)

    # Print the predicted probabilities
    print("Test Probabilities:", test_probabilities)

    # Compare the predicted probabilities to the true labels and print the comparison
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = data
            labels = torch.tensor([label[1] == 1 for label in labels])
            outputs = net(inputs)
            hater_probabilities = [output[1] for output in outputs]
            print(roc_auc_score(labels, hater_probabilities))
            _, predicted = torch.max(outputs.data, 1)
            print(f1_score(labels, predicted))
            total += labels.size(0)
            correct += (predicted == labels).sum().item() if predicted.numel() > 1 else (predicted == labels).item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test inputs: %d %%' % accuracy)

    # Write the results to a text file
    with open('/nas/home/jfu/data/results.txt', 'w') as f:
        f.write('Accuracy: %d %%\n' % accuracy)
        f.write('F1 Score: %f\n' % f1_score(labels, predicted))
        f.write('ROC AUC Score: %f\n' % roc_auc_score(labels, hater_probabilities))

if __name__ == "__main__":
    print("Hello")
    main()
    print("end statement")
