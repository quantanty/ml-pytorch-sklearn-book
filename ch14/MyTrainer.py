import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_epoch(self, device, log):
        loss_log = 0
        acc_log = 0
        self.model.train()
        for x, y in self.train_loader:
            x = x.to(device)
            y = y.to(device).float()

            preds = self.model(x).squeeze(1)
            loss = self.loss_fn(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_log += loss.item() * y.size(0)
            correct = ((preds>=0.5) == y).float()
            acc_log += correct.sum().cpu()

        if log:
            return loss_log / len(self.train_loader.dataset), acc_log / len(self.train_loader.dataset)

    def validate(self, device, log):
        loss_log = 0
        acc_log = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(device)
                y = y.to(device).float()

                preds = self.model(x).squeeze(1)
                loss = self.loss_fn(preds, y)

                loss_log += loss.item() * y.size(0)
                correct = ((preds>=0.5) == y).float()
                acc_log += correct.sum().cpu()

        if log:
            return loss_log / len(self.valid_loader.dataset), acc_log / len(self.valid_loader.dataset)
        
    def train(self, epochs, scheduler=None, log=False, device=torch.device('cpu')):
        loss_log = [0] * epochs
        acc_log = [0] * epochs
        val_loss_log = [0] * epochs
        val_acc_log = [0] * epochs

        for epoch in range(epochs):
            loss_log[epoch], acc_log[epoch] = self.train_epoch(device, log)
            val_loss_log[epoch], val_acc_log[epoch] = self.validate(device, log)
            if scheduler:
                scheduler.step()

            print(f'Epoch {epoch+1} acc: {acc_log[epoch]:.4f} val_acc: {val_acc_log[epoch]:.3f}')
            

        if log:
            return loss_log, acc_log, val_loss_log, val_acc_log