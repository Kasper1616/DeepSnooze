from deepsnooze.models.ffnn import SimpleFFNN
from deepsnooze.data_cached import SleepyRatCachedDataset
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch


log_interval = 10
batch_size = 16

def train(net, trainloader, criterion, optimizer):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == log_interval - 1:    # print every log_interval mini-batches
                current_samples = (i + 1) * batch_size
                total_samples = len(trainloader.dataset)
                print(f'[Epoch {epoch + 1}, Batch {i + 1:3d}] '
                f'Samples: {current_samples}/{total_samples} | '
                f'Loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        
        correct = 0
        total = 0
        net.eval() # Set model to evaluation mode (turns off dropout, etc.)
        
        with torch.no_grad(): # Don't calculate gradients for validation (saves memory)
            for data in valloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1) # Get the class with highest probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'>>> End of Epoch {epoch + 1} | Validation Accuracy: {accuracy:.2f}%')
        print("-" * 30)

if __name__ == "__main__":
    import numpy as np
    # Example usage
    full_dataset = SleepyRatCachedDataset(processed_path="data_processed/")

    # Create a subset
    indices = np.random.choice(len(full_dataset), size=10000, replace=False)
    subset = Subset(full_dataset, indices)

    train_size = int(0.8 * len(subset))
    test_size = len(subset) - train_size
    train_dataset, val_dataset = random_split(subset, [train_size, test_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size,
                                              shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    model = SimpleFFNN()
    train(net=model,
          trainloader=trainloader,
          criterion=torch.nn.CrossEntropyLoss(),
          optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9))