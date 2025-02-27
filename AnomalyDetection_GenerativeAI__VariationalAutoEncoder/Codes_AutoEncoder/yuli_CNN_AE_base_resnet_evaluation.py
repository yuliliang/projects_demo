import torch
import heapq
from torch import nn
import matplotlib.pyplot as plt
import yuli_CNN_AE_base_resnet_config as cfg
from torch.utils.data import DataLoader
import numpy as np
device = cfg.device


class EvalModel:
    def __init__(self, device: str):
        self.device = device

    def eval_model(self, model: nn.Module, test_loader):
        """
        Goal: Evaluate the model by the test data loss.

        Args:
            model: The saved model
            test_loader: Dataloader of test data.

        Returns:
            Currently showing the loss in the console instead of return the values.
        """
        model.eval()
        test_loss = []
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_loader, 1):
                dec_output, enc_input = model.forward(batch['image'].to(device, dtype=torch.float32))
                loss = model.loss_function(dec_output, enc_input, model, cfg.L1_reg_coef, cfg.L2_reg_coef)
                test_loss.append(loss.item())

        avg_test_loss = sum(test_loss) / len(test_loss)
        print('Avg. Validation loss = ', '{:.7f}'.format(avg_test_loss))
        model.train()
        return 1

    def Sampling_from_latent_space(self, model):
        model.eval()
        samples = model.sample(num_samples=15, device=device)
        samples = samples.detach().cpu().numpy()
        samples = np.squeeze(samples)
        # show reconstructions
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 7))
        k=0
        for i in range(3):
            for j in range(5):
                axes[i][j].imshow(samples[k], cmap='gray')
                axes[i][j].axis('off')
                k=k+1
        plt.tight_layout(pad=0.)
        plt.savefig(cfg.DIR_out + 'check/' +'AE_sample_from_latent_space.png')
        plt.close()
        return 1

    def Reconstruction(self, model, all_dataset, L_idx):
        fig, axes = plt.subplots(nrows=2, ncols=len(L_idx), figsize=(10, 5))
        plt.suptitle("Top: Original images. Bottom: Reconstructed images.")
        for j in range(len(L_idx)):
            original_img = all_dataset.img_show(L_idx[j])
            tensor_img = all_dataset.img_tensor(L_idx[j])
            recons = model.generate(tensor_img.to(device))

            axes[0][j].imshow(original_img, cmap='gray')
            axes[0][j].axis('off')
            axes[1][j].imshow(np.squeeze(recons).detach().cpu().numpy(), cmap='gray')
            axes[1][j].axis('off')

        plt.savefig(cfg.DIR_out + 'check/' +'AE_imge_reconstruction.png')
        plt.close()
        return 1

    def Interpolation(self, model, all_dataset, idx_img1, idx_img2):
        original_img_1 = all_dataset.img_show(idx_img1)
        original_img_2 = all_dataset.img_show(idx_img2)

        starting_inputs = all_dataset.img_tensor(idx_img1)
        ending_inputs = all_dataset.img_tensor(idx_img2)

        granularity = 10
        interpolation = model.interpolate(
            starting_inputs=starting_inputs, ending_inputs=ending_inputs, device=device,
            granularity=granularity).squeeze()
        fig, axes = plt.subplots(nrows=1, ncols=granularity + 2, figsize=(20, 5))

        for j in range(granularity):
            axes[j + 1].imshow(np.squeeze(interpolation[j].detach().cpu().numpy()), cmap='gray')
            axes[j + 1].axis('off')

        axes[0].imshow(np.squeeze(original_img_1), cmap='gray')
        axes[0].axis('off')
        axes[granularity + 1].imshow(np.squeeze(original_img_2), cmap='gray')
        axes[granularity + 1].axis('off')

        plt.tight_layout(pad=0.)
        plt.savefig(cfg.DIR_out + 'check/' + 'AE_img_interpolation_sample.png')
        plt.close()

    def anomaly_detection(self, model, all_dataset):
        n_sample = 5

        full_loader = DataLoader(dataset=all_dataset, shuffle=True, batch_size=cfg.batch_size_train,
                                 num_workers=cfg.dl_N_workers, pin_memory=cfg.PinM_TF)
        model.eval()
        recon_errors = np.array([])
        L_idx = np.array([])
        L_img_name =[]
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(full_loader, 1):
                L_idx = np.concatenate([L_idx, batch['idx'].numpy()])
                L_img_name += batch['img_name']
                dec_output, enc_input = model.forward(batch['image'].to(device, dtype=torch.float32))
                recon_errors = np.concatenate(
                    [recon_errors, torch.mean((dec_output - enc_input) ** 2, dim=[1, 2, 3]).numpy()])

        percentile_thr = 99
        threshold = np.percentile(recon_errors, percentile_thr)  # Set threshold at 95th percentile
        anomalies = recon_errors > threshold

        error_and_img_idx = [(recon_errors[i], L_idx[i]) for i, v in enumerate(recon_errors)]  #  if v==True ]
        top_big_n = heapq.nlargest(2*n_sample, error_and_img_idx)
        anomalies_idx = [int(index) for value, index in top_big_n]  # [(index, value) for value, index in top_n]

        normal = recon_errors <= threshold
        normal_idx = L_idx[normal].astype(int)

        print(f"Anomaly count: {sum(anomalies)}")
        plt.hist(recon_errors, bins=50, alpha=0.7)
        plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Frequency")
        plt.title("Anomaly Detection using AE (" + str(percentile_thr) + " % )")
        plt.savefig(cfg.DIR_out + 'check/' + 'AE_anomaly_detection_hist.png')
        plt.close()

        fig, axes = plt.subplots(nrows=4, ncols=n_sample, figsize=(10, 10))
        plt.suptitle("Top 10: normal. Bottom 10: abnormal (threshold: " + str(percentile_thr) + " percentile)")
        k = 0
        for i in [0, 1]:
            for j in range(n_sample):
                axes[i][j].imshow(all_dataset.img_show(normal_idx[k]), cmap='gray')
                axes[i][j].axis('off')
                k += 1
        k = 0
        for i in [2, 3]:
            for j in range(n_sample):
                axes[i][j].imshow(all_dataset.img_show(anomalies_idx[k]), cmap='gray')
                axes[i][j].axis('off')
                k += 1

        plt.tight_layout(pad=0.)
        plt.savefig(cfg.DIR_out + 'check/' +'AE_anomaly_detection_samples.png')
        plt.close()
        return 1