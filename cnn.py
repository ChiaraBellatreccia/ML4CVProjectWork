import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm

device = 'cuda'

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512 * 8 * 8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 8 * 8)
        x = self.fc(x)
        return x
    
def train(net, FLAG, EPOCHS, LEARNING_RATE, BATCH_SIZE, optimizer, criterion, scheduler, early_stopper, train_loader, validation_loader):
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        train_loss = 0.0
        val_loss = 0.0
        running_loss = 0.0

        all_train_preds = []
        all_train_labels = []

        net.train()

        for i, data in tqdm(enumerate(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # store loss for plotting later
            train_loss += loss.item()
            running_loss += loss.item()

            # collect predictions and true labels for F1 score calculation
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

            # print statistics
            if i % 1000 == 0:    # print every 1000 mini-batches
                if i == 0:
                  print(f'[Epoch {epoch + 1}, Batch {i + 1:5d} / {len(train_loader)}] Training Loss: {running_loss}', flush=True)
                else:
                  print(f'[Epoch {epoch + 1}, Batch {i + 1:5d} / {len(train_loader)}] Training Loss: {running_loss / 1000:.3f}', flush=True)
                  running_loss = 0.0

        # Calculate training F1 score for the epoch
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        train_f1_scores.append(train_f1)

        train_losses.append(train_loss / len(train_loader))

        net.eval()  # Set the network to evaluation mode

        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():  # Disable gradient calculation for validation
            for i, data in tqdm(enumerate(validation_loader, 0)):
                inputs, _, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # collect predictions and true labels for F1 score calculation
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate validation F1 score for the epoch
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_f1_scores.append(val_f1)
        val_losses.append(val_loss / len(validation_loader))

        print(f'[Epoch {epoch + 1}, Validation Loss: {val_loss / len(validation_loader):.3f}]', flush=True)

        # save the model if validation loss is lower than previous epoch
        if val_f1_scores[epoch] >= val_f1_scores[epoch-1]:
            torch.save(net.state_dict(), f'CNN/weights/epcs_{EPOCHS}__lr_{LEARNING_RATE}__bs_{BATCH_SIZE}__{FLAG}.pth')

        # early stopping
        if early_stopper is not None:
          if early_stopper.early_stop(val_losses[epoch]):
            torch.save(net.state_dict(), f'CNN/weights/epcs_{EPOCHS}__lr_{LEARNING_RATE}__bs_{BATCH_SIZE}_LASTEPOCH__{FLAG}.pth')
            break

        # cosine annealing step
        if scheduler is not None:
          scheduler.step()
          current_lr = scheduler.get_last_lr()[0]
          print('UPDATING LEARNING RATE TO: ', current_lr, flush=True)

    # save loss and F1 score plots to file
    # Plotting the training and validation loss
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(train_f1_scores, label='Training F1 Score')
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'CNN/plots/epcs_{EPOCHS}__lr_{LEARNING_RATE}__bs_{BATCH_SIZE}__{FLAG}.png')

    print('Finished Training', flush=True)

    torch.save(net.state_dict(), f'CNN/weights/epcs_{EPOCHS}__lr_{LEARNING_RATE}__bs_{BATCH_SIZE}__{FLAG}_LASTEPOCH.pth')

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False