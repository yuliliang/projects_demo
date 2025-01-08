import torch
from torch import nn
import matplotlib.pyplot as plt
import yuli_CNN_AE_base_resnet_config as cfg
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torchvision.transforms as transforms
device = cfg.device


class EvalModel:
    def __init__(self, device: str):
        self.device = device

    def eval_model(self, model: nn.Module, Loss_main, Loss_1_2, test_loader):
        """
        Goal: Evaluate the model by the test data loss.

        Args:
            model: The saved model
            Loss_main: Loss function
            Loss_1_2: Loss from L1 & L2 regularization.
            test_loader: Dataloader of test data.

        Returns:
            Currently showing the loss in the console instead of return the values.
        """
        model.eval()
        test_loss = []
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_loader, 1):
                dec_output = model(batch['img'].to(cfg.device, dtype=torch.float32))
                loss = Loss_main(dec_output, batch['img'].to(cfg.device, dtype=torch.float32)) \
                       + Loss_1_2.loss_function(model)
                test_loss.append(loss.item())

        avg_test_loss = sum(test_loss) / len(test_loss)
        print('Avg. Validation loss = ', '{:.7f}'.format(avg_test_loss))
        model.train()
        return 1

    def create_embedding(self, AE, full_loader):   # full_loader
        """
        Goal: With data from full_loader, creates embedding using encoder of the trained AutoEncoder.

        Args:
            AE: AutoEncoder model
            full_loader: PyTorch dataloader that contains (images, images) over entire dataset.

        Returns:
            embedding: Image embedding of size (num_images_in_loader + 1, c, h, w)
            embedding_FName: The file name of the original image, for each corresponding embeddings.
            List_article_id: The corresponding article_id or each image embedding.
        """
        AE.eval()  # Set encoder to eval mode, since we do not change model here.
        # Just a place holder for our 0th image embedding.
        embedding = []  # torch.randn(embedding_dim)

        with torch.no_grad():
            # for batch_idx, (train_img, target_img) in enumerate(full_loader):
            for (batch_idx, batch) in enumerate(full_loader, 1):  # enumerate(**, 1): make the batch index start from "1"
                # Get encoder outputs and move outputs to cpu
                enc_output = AE.forward_encoder(batch['img'].to(device, dtype=torch.float32))
                List_FName = batch['img_FName']

                # Keep append outputs to embeddings results.
                if embedding == []:  # if no embedding, initialize it with correct embedding size
                    cfg.embed_channel = enc_output.shape[1]
                    cfg.embed_h = enc_output.shape[2]
                    cfg.embed_w = enc_output.shape[3]
                    embedding_dim = (1, enc_output.shape[1], enc_output.shape[2], enc_output.shape[3])

                    embedding = torch.randn(embedding_dim)
                    embedding = torch.cat((embedding, enc_output), 0)
                    embedding_FName = ['']
                    embedding_FName += List_FName
                    List_article_id = [-1]
                    List_article_id += [int(FName.split("/")[-1][1:-4]) for FName in List_FName]
                else:
                    embedding = torch.cat((embedding, enc_output), 0)
                    embedding_FName += List_FName
                    List_article_id += [int(FName.split("/")[-1][1:-4]) for FName in List_FName]

        return embedding, embedding_FName, List_article_id

    def knn_image_search(
            self, all_dataset, img_idx, num_images, embeddings, embedding_FName, model):
        """
        Goal: Search knn among the image embeddings.

        Args:
            all_dataset: The image dataset.
            img_idx: The index of one query image.
            num_images: Number of the nearest neighbors would like to search.
            embeddings: all the embeddings we'd like to search.
            embedding_FName: The corresponding file name of each embedding.
            model: The image model. Here is The Auto-Encoder.

        Returns:
            No return. Here we save the images of knn results.
        """
        # ...... get query image embedding ........
        img_FName = all_dataset.annotations.iloc[img_idx, 0]
        img_raw = Image.open(img_FName).convert("RGB")   # os.path.join(all_dataset.root_dir,
        image_tensor = all_dataset.transform(img_raw)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)  # create a batch_size = 1
        with torch.no_grad():
            image_embedding = model.forward_encoder(image_tensor.to(device, dtype=torch.float32))
        flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

        # ...... get top n nearest neighbor by embeddings ........
        knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
        knn.fit(embeddings.reshape(embeddings.shape[0], -1))

        _, indices = knn.kneighbors(flattened_embedding)  # return neigh_dist & neighor_index
        List_knn_index = indices.tolist()[0]

        # ...... plot the query & KNN images .....
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        imgs = [transform_to_tensor(img_raw).permute(1,2,0)]
        for idx in List_knn_index:
            FName = embedding_FName[idx]
            img_raw = transform_to_tensor(Image.open(FName).convert("RGB"))
            imgs.append(img_raw.permute(1,2,0))

        f, ax = plt.subplots(1, len(List_knn_index)+1, figsize=(24, 12))
        for idx, img in enumerate(imgs):
            ax[idx].imshow(img)
            ax[idx].axis('off')
            if idx == 0:
                ax[idx].set_title('Query', fontsize=40)
            elif idx == 1:
                ax[idx].set_title('1st nn', fontsize=40)
            elif idx == 2:
                ax[idx].set_title('2nd nn', fontsize=40)
            elif idx == 3:
                ax[idx].set_title('3rd nn', fontsize=40)

        plt.savefig(cfg.DIR_out + 'check/' +'nearest_neighbor_of_img_' + str(img_idx) + '.png')
        plt.close()
        return 1

    def display_reconstructed_img(self, AE, all_dataset, img_idx):
        """
        Goal: With a saved Auto-Encoder, reconstruct some image to check what the AE learned.

        Args:
            AE: A saved Auto-Encoder Model.
            all_dataset: The input image dataset.
            img_idx: the index of the input image.

        Returns:
            No return. Here we save an image of "original image vs. reconstructed image".
        """
        img = (Image.open(all_dataset.annotations.iloc[img_idx, 0]).convert("RGB"))

        AE.eval()
        with torch.no_grad():
            rec_img = AE.forward((all_dataset.transform(img)).unsqueeze(0))  # make the batch_size=1 here.
        rec_img = rec_img.cpu().squeeze()  # rec_img -> (1, 3, xx, xx) but img.squeeze() -> (3,xx,xx)

        f, ax = plt.subplots(1, 2, figsize=(24, 12))
        transform_imshow = transforms.Compose([transforms.ToTensor()])
        ax[0].imshow(transform_imshow(img).permute((1, 2, 0)))
        ax[0].axis('off')
        ax[0].set_title(str(0) + ':query', fontsize=40)

        ax[1].imshow(rec_img.permute((1, 2, 0)))
        ax[1].axis('off')
        ax[1].set_title(str(1) + ': reconstructed image', fontsize=40)

        plt.savefig(cfg.DIR_out + 'check/' + 'reconstructed_image_of_' + str(img_idx) + '.png')
        plt.close()
        return 1

        # def plot_vline_at_x_value(self, series_x_values, series_y_values, pick_color):
        #     y_values = list(series_y_values)
        #     for i in series_x_values:
        #         plt.vlines(i, min(y_values), max(y_values), colors=pick_color, linestyles='solid')
