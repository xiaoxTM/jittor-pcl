from jittor import nn
from sigma.nn.jittor import losses,optimizers,schedulers
from sigma.reporters import ProgressBar
from sigma.fontstyles import colors
import argparse
import os.path

class BasicModule(nn.Module):
    def __init__(self,args,margs):
        super(BasicModule,self).__init__()
        self.loss = self.build_losses(args.loss)

    @classmethod
    def build_model(cls,args,margs):
        raise NotImplementedError()

    def build_dataloaders(self,args,margs,splits='both',shuffle=True):
        raise NotImplementedError()

    def train_epoch(self,dataset,bar,epoch):
        raise NotImplementedError()

    def trial_epoch(self,dataset,bar,epoch,cache_size):
        raise NotImplementedError()

    def valid_epoch(self,dataset,bar,epoch):
        raise NotImplementedError()

    def build_optimizers(self,optimizer,scheduler,lr):
        self.optimizer = optimizers.get(optimizer,lr[0],self.parameters())
        self.scheduler = schedulers.get(scheduler,self.optimizer)

    def build_losses(self,loss):
        return losses.get(loss)

    @classmethod
    def get_argument_parser(cls):
        parser = argparse.ArgumentParser()
        subparser = parser.add_subparsers()
        # train
        subparser_train = subparser.add_parser('train')
        subparser_train.set_defaults(func=lambda args,margs:cls.run_train(args,margs))
        subparser_train.add_argument('--exp-name',type=str,default=None)
        subparser_train.add_argument('--optimizer',type=str,default='sgd(momentum=0.9,weight_decay=1e-4)')
        subparser_train.add_argument('--scheduler',type=str,default=None)
        subparser_train.add_argument('--lr',type=float,default=[0.1],nargs='+')
        subparser_train.add_argument('--checkpoints',type=str,default='runs')
        subparser_train.add_argument('--epochs',type=int,default=100)
        subparser_train.add_argument('--begin-valid-epoch',type=int,default=10)
        # trial
        subparser_trial = subparser.add_parser('trial')
        subparser_trial.set_defaults(func=lambda args,margs:cls.run_trial(args,margs))
        subparser_trial.add_argument('--optimizer',type=str,default='sgd(momentum=0.9,weight_decay=1e-4)')
        subparser_trial.add_argument('--scheduler',type=str,default=None)
        subparser_trial.add_argument('--lr',type=float,default=[0.1],nargs='+')
        subparser_trial.add_argument('--cache-size',type=int,default=1)
        subparser_trial.add_argument('--epochs',type=int,default=100)
        # test
        subparser_test = subparser.add_parser('test')
        subparser_test.set_defaults(func=lambda args,margs:cls.run_test(args,margs))

        return parser,(subparser_train,subparser_trial,subparser_test)

    @classmethod
    def run_train(cls,args,margs):
        stamp, logger = cls.init(args,margs)
        net = cls.build_model(args,margs)
        net.build_optimizers(margs.optimizer,margs.scheduler,margs.lr)
        net.build_dataloaders(args,margs,'both')
        net._train(args,margs,stamp,logger)
        logger.close()

    def _train(self,args,margs,stamp,logger):
        bar = ProgressBar(margs.epochs, keep_line=False)
        train_bar = bar.sub(self.train_set, keep_line=False,cc='>')
        valid_bar = bar.sub(self.valid_set, keep_line=False,cc='=')
        for epoch in bar:
            logger.train()
            self.train_epoch(train_bar,epoch)
            if (epoch+1) >= margs.begin_valid_epoch:
                logger.eval()
                need_save_model = self.valid_epoch(valid_bar,epoch)
                if need_save_model:
                    ckpt = os.path.join(margs.checkpoints,stamp,'model.pkl')
                    self.save(ckpt)
            if self.scheduler is not None:
                self.scheduler.step()

    @classmethod
    def run_trial(cls,args,margs):
        net = cls.build_model(args,margs)
        net.build_optimizers(margs.optimizer,margs.scheduler,margs.lr)
        net.build_dataloaders(args,margs,'train',shuffle=False)
        net._trial(args,margs)

    def _trial(self,args,margs):
        from datasets.utils import subset
        bar = ProgressBar(margs.epochs, keep_line=False)
        sub_train_set = subset(self.train_set,margs.cache_size)
        train_bar = bar.sub(sub_train_set, keep_line=False)
        for epoch in bar:
            self.trial_epoch(train_bar,epoch)
            if self.scheduler is not None:
                self.scheduler.step()

    @classmethod
    def run_test(cls,args,margs):
        net = cls.build_model(args,margs)
        net.build_dataloaders(args,margs,'test')
        net._test(args,margs)

    def _test(self,args,margs):
        bar = ProgressBar(self.test_set,cc='=')
        self.valid_epoch(bar)

    @classmethod
    def init(cls,args,margs):
        from sigma.utils import timestamp
        from sigma.ios import backups
        from sigma.reporters import Logger,save_configs
        stamp = timestamp()
        if margs.exp_name is not None:
            stamp = '{}-{}'.format(margs.exp_name,stamp)
        print('experiment name:',colors.red(stamp))
        os.makedirs(os.path.join(margs.checkpoints,stamp),exist_ok=True)
        save_configs(os.path.join(margs.checkpoints,stamp,'config'),[args,margs])
        logger = Logger(os.path.join(margs.checkpoints,stamp,'run.log'))
        return stamp, logger
