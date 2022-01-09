import pandas as pd
import numpy as np
import random
import os
import nltk
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.legacy.data import Field, LabelField, BucketIterator, TabularDataset
from torchtext import vocab
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import spearmanr
from nltk import word_tokenize

seed = 42
source_folder = 'Data'
num_epochs = 2


class LSTM_model(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim=105, hidden_size=179, output_dim=1, dropout_rate=0.58,
                 **kwargs):

        super(LSTM_model, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, **kwargs)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.linear = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, tensor_batch):

        embedding_tensor = self.embedding(tensor_batch)

        dropout_embedding = self.dropout(embedding_tensor)

        out, (hidden_state, _) = self.lstm(dropout_embedding)

        hidden_squeezed = hidden_state.squeeze(0)

        assert torch.equal(out[-1, :, :], hidden_squeezed)

        return self.linear(hidden_squeezed)

def get_auroc(truth, pred):
    assert len(truth) == len(pred)
    auc = roc_auc_score(truth.numpy(), pred.numpy())
    return auc

def spearman(x,y):
    return spearmanr(x,y)

def train_model(model, train_iter, optimizer):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in train_iter:

        optimizer.zero_grad()

        predictions = model(batch.comment).squeeze(1) # removing the extra dimension ([batch_size,1])

        loss_function = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([20394 / 2753] * len(batch)))

        loss = loss_function(predictions, batch.attack)  # batch loss

        predicted_classes = torch.round(torch.sigmoid(predictions))

        correct_preds = (predicted_classes == batch.attack).float()

        accuracy = correct_preds.sum() / len(correct_preds)


        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # add the loss for this batch to calculate the loss for whole epoch
        epoch_acc += accuracy.item()  # .item() tend to give the exact number from the tensor of shape [1,]



    return epoch_loss / len(train_iter), epoch_acc / len(train_iter)

def evaluate_model(model, val_test_iter, optimizer):

    total_loss = 0
    total_acc = 0

    # Two lists are used to calculate AUC score
    y_true = []
    y_pred = []
    y_pred_round = []

    model.eval()

    with torch.no_grad():

        for batch in val_test_iter:
            predictions = model(batch.comment).squeeze(1)

            loss_function = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([20392 / 2755] * len(batch)))

            loss = loss_function(predictions, batch.attack)

            predicted_classes = torch.sigmoid(predictions)
            y_pred.append(predicted_classes)

            pred_classes = torch.round(torch.sigmoid(predictions))
            y_pred_round.append(pred_classes)

            correct_predictions = (pred_classes == batch.attack).float()

            accuracy = correct_predictions.sum() / len(correct_predictions)

            total_loss += loss.item()
            total_acc += accuracy.item()
            y_true.append(batch.attack)

        return total_loss / len(val_test_iter), total_acc / len(val_test_iter), y_pred, y_true, y_pred_round

if __name__ == '__main__':

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.determinstic = True

    os.environ['PYTHONHASHSEED'] = str(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use 'cuda' if available else 'cpu

    df_train = pd.read_csv(source_folder+'/train.csv', sep=',')
    df_valid = pd.read_csv(source_folder+'/valid.csv', sep=',')
    df_test = pd.read_csv(source_folder+'/test.csv', sep=',')

    comment_field = Field(tokenize=word_tokenize)
    # tokenize text using word_tokenize and convert to numerical form using default parameters

    label_field = LabelField(dtype=torch.float)
    # useful for label string to LabelEncoding. Not useful here but doesn't hurt either

    fields = [('comment', comment_field), ('attack', label_field)]
    # (column name,field object to use on that column) pair for the dictonary

    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv', test='test.csv',
                                             format='csv', skip_header=True, fields=fields)

    comment_field.build_vocab(train)
    label_field.build_vocab(train)

    train_iter, valid_iter, test_iter = BucketIterator.splits((train, valid, test), batch_sizes=(128, 512, 512),
                                                            sort_key=lambda x: len(x.comment),
                                                            sort_within_batch=False,
                                                            device=device)  # use the cuda device if available
    vocab_size = len(comment_field.vocab)
    lr = 3e-4  # learning rate = 0.0003

    model = LSTM_model(vocab_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_iter, optimizer)
        validation_loss, validation_acc, y_pred, y_true, test_y_pred = evaluate_model(model, valid_iter, optimizer)
        train_loss_list.append(train_loss)
        valid_loss_list.append(validation_loss)
        print(
            f'''End of Epoch: {epoch + 1}  |  Train Loss: {train_loss:.3f}  |  Validation Loss: {validation_loss:.3f}  |  Train Acc: {train_acc * 100:.2f}%  |  Validation Acc: {validation_acc * 100:.2f}% ''')

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    sns.lineplot(x=np.arange(1,num_epochs+1), y=train_loss_list, color='blue', label='Train');
    sns.lineplot(x=np.arange(1, num_epochs + 1), y=valid_loss_list, color='orange', label='Valid');
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("test.png")

    test_loss, test_acc, test_y_pred, test_y_true, test_y_pred_round = evaluate_model(model, test_iter, optimizer)
    test_y_pred_cat = torch.cat(test_y_pred)
    test_y_true_cat = torch.cat(test_y_true)
    test_auc = get_auroc(test_y_true_cat, test_y_pred_cat)
    test_spear = spearman(test_y_true_cat,test_y_pred_cat)
    print(f'''Test AUC score: {test_auc:.3f}''')
    print(test_spear)
