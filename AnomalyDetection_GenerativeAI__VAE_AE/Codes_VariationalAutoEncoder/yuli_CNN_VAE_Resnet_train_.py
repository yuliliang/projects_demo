import torch, os, time
import yuli_CNN_VAE_Resnet_config as cfg
device = cfg.device


# ========== regular Training process ===========
def train_with_early_stop(writer, model, epochs, optimizer, scheduler_lr, train_loader,
                          validate_loader, str_out, DIR_out, L_norm_coef, PARA, epoch_start=1):
    """
    Goal: Train a NN model with early stop.

    Args:
        writer: Tensorboard writer.
        model: Initial model before training.
        epochs: Total epoch.
        optimizer:
        scheduler_lr: Learning rate scheduler.
        train_loader: Train data loader.
        validate_loader: Validation data loader.
        str_out: The string that denote the model information, and will put in the output files.
        DIR_out: Folder for the output file.
        L_norm_coef: A list of coefficients when pre-processing the data.
        PARA: Model parameters.
        epoch_start: starting index of the epoch.

    Returns:
        None. The trained model would be saved as "PATH_model".
    """
    train_start_T = time.time()

    # parameters for early stopping
    valid_loss_min = 10000
    epoch_min = -1
    patience = 10
    early_stop_trigger_times = 0
    print_batch = 5000

    DIR_model = 'model/'
    if not os.path.isdir(DIR_out + DIR_model):
        os.mkdir(DIR_out + DIR_model)
    PATH_model = DIR_out + DIR_model + 'model' + str_out

    List_epoch_loss_train = []
    List_epoch_loss_valid = []

    step = -1
    for epoch in range(epoch_start, epoch_start + epochs):
        step += 1
        model.train()
        print("\n=============== Epoch " + str(epoch) + " ===============")

        # ............ Training for each epoch ............
        train_loss = []

        for (batch_idx, batch) in enumerate(train_loader, 1):  # enumerate(**, 1): make the batch index start from "1"
            optimizer.zero_grad(set_to_none=True)
            dec_output, enc_input, mu, log_var, _ = model.forward(batch['image'].to(device, dtype=torch.float32))
            loss = model.loss_function(dec_output, enc_input, mu, log_var)
            loss.backward()
            optimizer.step()
            scheduler_lr.step()
            train_loss.append(loss.item())

            if batch_idx % print_batch == 0:
                print("Batch # = " + str(batch_idx) + ', Epoch:', '%04d' % (epoch), 'loss =', '{:.4f}'.format(loss))

        avg_train_loss = sum(train_loss) / len(train_loss)
        List_epoch_loss_train.append(avg_train_loss)
        print('Epoch:', '%03d' % (epoch), ', Avg. Train loss = ', '{:.5f}'.format(avg_train_loss))
        writer.add_scalar("Loss/train", avg_train_loss, global_step=step)

        # ...... evaluate this epoch loss by validation data ........
        model.eval()
        valid_loss = []
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(validate_loader, 1):
                dec_output, enc_input, mu, log_var, _ = model.forward(batch['image'].to(device, dtype=torch.float32))
                loss = model.loss_function(dec_output, enc_input, mu, log_var)
                valid_loss.append(loss.item())

        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        List_epoch_loss_valid.append(avg_valid_loss)
        print('Epoch:', '%03d' % (epoch), ', Avg. Validation loss = ', '{:.7f}'.format(avg_valid_loss))
        writer.add_scalar("Loss/valid", avg_valid_loss, global_step=step)

        # ........ Early stopping ..........
        if valid_loss_min > avg_valid_loss:
            valid_loss_min = avg_valid_loss
            print('early_stop_trigger_times: 0')
            early_stop_trigger_times = 0
            epoch_min = epoch

            # update parameters for the saved model
            PARA['epoch'] = epoch
            PARA['epoch_min'] = epoch_min
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            scheduler_lr_state_dict = scheduler_lr.state_dict()

        else:
            early_stop_trigger_times += 1
            print('early_stop_trigger_times:', early_stop_trigger_times)

            if early_stop_trigger_times >= patience:
                # ........ save model ......
                PARA['epoch'] = epoch
                PARA['train_time_in_minute'] = (time.time() - train_start_T) / 60
                torch.save({
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                    'scheduler_lr_state_dict': scheduler_lr_state_dict,
                    'L_norm_coef': L_norm_coef,  # A list of coefficients when pre-processing the data.
                    'PARA': PARA  # Model parameters
                }, PATH_model)

                return 1
        model.train()
    # ................................................
    PARA['epoch'] = epoch
    PARA['train_time_in_minute'] = (time.time() - train_start_T) / 60
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_lr_state_dict': scheduler_lr_state_dict,
        'L_norm_coef': L_norm_coef,  # A list of coefficients when pre-processing the data.
        'PARA': PARA  # model parameters
    }, PATH_model)

    return 1
