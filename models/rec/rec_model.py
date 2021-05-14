from jittor import nn
import jittor as jt
from models.basic_model import BasicModule
import numpy as np
from sklearn import metrics
from datetime import datetime

class RecModule(BasicModule):
    def __init__(self,args,margs):
        super(RecModule,self).__init__(args,margs)
        self.best_acc = 0

    def build_dataloaders(self,args,margs,splits='both',shuffle=True):
        from datasets.modelnet40 import ModelNet40
        if splits == 'train':
            self.train_set = ModelNet40(args.dataset_root,
                                                                      args.num_points,
                                                                      splits,
                                                                      batch_size=args.batch_size,
                                                                      shuffle=shuffle,
                                                                      drop_last=True,
                                                                      num_workers=args.num_workers)
        elif splits == 'test':
            self.test_set = ModelNet40(args.dataset_root,
                                                                    args.num_points,
                                                                    splits,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=False,
                                                                    drop_last=False,
                                                                    num_workers=args.num_workers)
        elif splits == 'both':
            self.train_set = ModelNet40(args.dataset_root,
                                                                      args.num_points,
                                                                      'train',
                                                                      batch_size=args.batch_size,
                                                                      shuffle=shuffle,
                                                                      drop_last=True,
                                                                      num_workers=args.num_workers)
            self.valid_set = ModelNet40(args.dataset_root,
                                                                      args.num_points,
                                                                      'test',
                                                                      batch_size=args.batch_size,
                                                                      shuffle=False,
                                                                      drop_last=False,
                                                                      num_workers=args.num_workers)

    def train_epoch(self,bar,epoch):
        self.train()
        trues = []
        preds = []
        running_loss = 0
        running_num = 0
        for data in bar:
            samples, labels = data
            logits = self(samples)
            loss = list(self.loss(logits,labels).values())[0]
            self.optimizer.step(loss)
            pred = np.argmax(logits.data,axis=1)
            preds.extend(pred)
            trues.extend(labels.data.flatten())
            acc = metrics.accuracy_score(trues,preds)
            running_loss += loss.data[0]
            running_num += labels.shape[0]
            loss = running_loss / running_num
            bar.set_message(f'loss:{loss:.4f}, acc:{acc:.4f}')

    def trial_epoch(self,bar,epoch):
        self.train()
        trues = []
        preds = []
        running_loss = 0
        running_num = 0
        # cache the dataset
        # we only use the first cache-size batch of samples
        # to observe whether the model overfits on them
        for idx,data in enumerate(bar):
            samples, labels = data
            beg = datetime.now()
            logits = self(samples)
            loss = list(self.loss(logits,labels).values())[0]
            end = datetime.now()
            ftime = end - beg
            beg = datetime.now()
            self.optimizer.step(loss)
            end = datetime.now()
            btime = end - beg
            pred = np.argmax(logits.data,axis=1)
            preds.extend(pred)
            trues.extend(labels.data.flatten())
            acc = metrics.accuracy_score(trues,preds)
            running_loss += loss.data[0]
            running_num += labels.shape[0]
            loss = running_loss / running_num
            bar.set_message(f'loss:{loss:.4f}, acc:{acc:.4f}, {ftime}/{btime}')

    def valid_epoch(self,bar,epoch):
        trues = []
        preds = []
        running_loss = 0
        running_num = 0
        with jt.no_grad():
            self.eval()
            acc = 0
            for data in bar:
                samples, labels = data
                logits = self(samples)
                loss = list(self.loss(logits,labels).values())[0]
                self.optimizer.step(loss)
                pred = np.argmax(logits.data,axis=1)
                preds.extend(pred)
                trues.extend(labels.data.flatten())
                acc = metrics.accuracy_score(trues,preds)
                running_loss += loss.data[0]
                running_num += labels.shape[0]
                loss = running_loss / running_num
                bar.set_message(f'loss:{loss:.4f}, acc:{acc:.4f}')
            if acc > self.best_acc:
                self.best_acc = acc
                return True
            return False
