import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *
from segformer_pipeline import *

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        if not self.args.predict:
            self._get_data()
            self._select_optimizer()
        else:
            self._get_val_data()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)

        self.model_path=self.args.model_path

        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        if self.args.predict:
            self._get_val_data()
        else:
            self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.data_mean, self.data_std = load_data(**config)

    def _get_val_data(self):
        config = self.args.__dict__
        config['is_train'] = False
        self.vali_loader, self.data_mean, self.data_std = load_data(**config)

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        best_model_path = self.path + '/' + 'checkpoint.pth'
        recorder = Recorder(verbose=True)

        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)

        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae = metric(preds, trues, 0, 1)
        print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        self.model.train()
        return total_loss

    def predict(self,args):
        self._get_val_data()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        inputs_lst, preds_lst ,last_frame_list= [], [],[]
        vali_pbar = tqdm(self.vali_loader)

        for i,batch_x in enumerate(vali_pbar):
            pred_y = self.model(batch_x.to(self.device))
            last_frame_y=pred_y[:,10,:,:,:]
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                batch_x,pred_y,last_frame_y], [inputs_lst, preds_lst,last_frame_list]))

        inputs = np.concatenate(inputs_lst, axis=0)
        preds = np.concatenate(preds_lst, axis=0)
        last_frames = np.concatenate(last_frame_list, axis=0)
        folder_path = self.path + '/predictions/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(osp.join(folder_path, 'inputs.npy'),inputs)
        np.save(osp.join(folder_path, 'preds.npy'),preds)
        np.save(osp.join(folder_path, 'last_frames.npy'),last_frames)
        
        Segformer_Module(args)
