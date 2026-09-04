import torch
from os import path as osp
from tqdm import tqdm

from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class HATMAEModel(SRModel):
    """HAT-MAE model for pre-training."""

    def feed_data(self, data):
        # For SR-MAE, lq is LR image, gt is HR image
        # For standard MAE, only lq (the original image) is needed
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else: # For standard MAE, gt is the same as lq
            self.gt = self.lq.clone()

    def optimize_parameters(self, current_iter):
        self.log_dict = {}
        self.optimizer_g.zero_grad()
        
        # Pass both gt (HR) and lq (LR) to the network
        self.loss, self.pred, self.mask = self.net_g(
            self.gt, 
            imgs_lr=self.lq, 
            mask_ratio=self.opt['network_g'].get('mask_ratio', 0.75)
        )

        self.loss.backward()
        self.optimizer_g.step()

        self.log_dict['l_total'] = self.loss.item()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.loss, self.pred, self.mask = self.net_g(
                self.gt, 
                imgs_lr=self.lq, 
                mask_ratio=self.opt['network_g'].get('mask_ratio', 0.75)
            )
        self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        self.metric_results = {'mae_loss': 0}

        use_pbar = self.opt['val'].get('pbar', False)
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # get image name
            if 'lq_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            else:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            if hasattr(self.net_g, 'unpatchify'):
                restored_img = self.net_g.unpatchify(visuals['result'])
                sr_img = tensor2img([restored_img])
            else:
                sr_img = tensor2img([visuals['result']])

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            self.metric_results['mae_loss'] += self.loss.item()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        self.metric_results['mae_loss'] /= (idx + 1)
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if tb_logger:
            tb_logger.add_scalar(f'metrics/{dataset_name}/mae_loss', self.metric_results['mae_loss'], current_iter)

    def get_current_visuals(self):
        out_dict = {}
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.pred.detach().cpu()
        return out_dict
