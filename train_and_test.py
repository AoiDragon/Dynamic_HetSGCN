import torch
from SGC_LSTM import SGC_LSTM
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class Trainer:
    def __init__(self, config, train_loader, val_loader, test_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SGC_LSTM(self.device, config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.micro_f1_list = []
        self.macro_f1_list = []
        print(self.device)

    def calculate_metrics(self, out, label_list):
        result = []
        for i in range(len(out)):
            out[i] = self.softmax(out[i].cpu())
            result.append(out[i].max(1)[1])

        micro_f1 = sum(f1_score(label_list[i].cpu().detach().numpy(), result[i], average='micro') for i in
                       range(len(out))) / len(out)
        macro_f1 = sum(f1_score(label_list[i].cpu().detach().numpy(), result[i], average='macro') for i in
                       range(len(out))) / len(out)
        self.micro_f1_list.append(micro_f1)
        self.macro_f1_list.append(macro_f1)

        print('Micro f1 = %f, Macro f1 = %f' % (micro_f1, macro_f1))

    def train_and_val(self, train_loader, val_loader):

        # for epoch in range(config.epoch_num):
        epoch = 0
        cnt = 0
        min_loss = 2
        while True:
            # Train
            epoch += 1
            print('\nEpoch %d:' % epoch)
            print('\nTraining...')
            self.macro_f1_list.clear()
            self.micro_f1_list.clear()

            for batch in train_loader:

                self.optimizer.zero_grad()
                labels = batch['y']
                label_list = []

                # Separate labels
                for i in range(0, labels.size()[0], 25):
                    mask = labels[i].ge(0)  # 筛去用于填充的-1
                    label = torch.masked_select(labels[i], mask)
                    label_list.append(label)

                # Predict
                out = self.model(batch)

                # Compute loss
                loss = sum(self.loss(out[i], label_list[i]) for i in range(len(label_list))) / len(label_list)

                # Update
                loss.backward()
                self.optimizer.step()
                print('loss = %f' % loss)

                # Calculate metrics
                self.calculate_metrics(out, label_list)

            print('Average training Micro f1 = %f, Macro f1 = %f' % (
                np.mean(self.micro_f1_list), np.mean(self.macro_f1_list)))

            # Validation
            print('\nValidating...')
            self.macro_f1_list.clear()
            self.micro_f1_list.clear()

            loss_list = []
            for batch in val_loader:
                labels = batch['y']
                label_list = []

                # Separate labels
                for i in range(0, labels.size()[0], 25):
                    mask = labels[i].ge(0)  # 筛去用于填充的-1
                    label = torch.masked_select(labels[i], mask)
                    label_list.append(label)

                # Predict
                out = self.model(batch)

                # Compute loss
                loss = sum(self.loss(out[i], label_list[i]) for i in range(len(label_list))) / len(label_list)
                print("loss = %f" % loss)
                loss_list.append(loss)
                # Calculate metrics
                self.calculate_metrics(out, label_list)

            print('Average validating Micro f1 = %f, Macro f1 = %f' % (
                np.mean(self.micro_f1_list), np.mean(self.macro_f1_list)))

            # Terminate training
            avg_loss = np.mean(loss_list)
            print("Average validating loss = %f" % avg_loss)
            if min_loss >= avg_loss:
                cnt += 1
                if cnt >= 10:
                    checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                                  'epoch': epoch}
                    localtime = time.localtime(time.time())
                    save_path = './trained_model_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + str(
                        localtime.tm_hour) + '.pkl'
                    torch.save(checkpoint, save_path)
                    return
            else:
                min_loss = avg_loss

    def test(self, dataloader):
        print("\nTesting")
        self.macro_f1_list.clear()
        self.micro_f1_list.clear()

        for batch in dataloader:

            labels = batch['y']
            label_list = []
            # Separate labels
            for i in range(0, labels.size()[0], 25):
                mask = labels[i].ge(0)  # 筛去用于填充的-1
                label = torch.masked_select(labels[i], mask)
                label_list.append(label)

            # Predict
            out = self.model(batch)

            # Calculate metrics
            self.calculate_metrics(out, label_list)

        print('Average testing Micro f1 = %f, Macro f1 = %f' % (
            np.mean(self.micro_f1_list), np.mean(self.macro_f1_list)))
