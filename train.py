from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from utils import SpeechDataset, train_valid_split, get_parameter_number, ValueWindow
from params import HParams
from VAD_T import VADModel, add_loss, prediction
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.nn as nn
import numpy as np
import os
import wandb
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
train_on_gpu = torch.cuda.is_available()

if (train_on_gpu):
    print('Training on GPU.')
    torch.backends.cudnn.benchmark = True
else:
    print('No GPU available, training on CPU.')

DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')
hparams = HParams()
_format = '%Y-%m-%d %H:%M:%S.%f'


def train_epoch(model, train_loader, loss_fn, optimizer, scheduler, batch_size, epoch, start_stpe):
    model.train()
    count = 0
    total_loss = 0
    step = start_stpe
    examples = []
    # mask = subsequent_mask(batch_size).unsqueeze(1).to(DEVICE)
    total_loss_window = ValueWindow(100)
    post_acc_window = ValueWindow(100)
    post_loss_window = ValueWindow(100)

    for x, y in train_loader:
        count += 1
        examples.append([x[0], y[0]])

        if count % 26 == 0:
            # examples.sort(key=lambda x: len(x[-1]))
            examples = (np.vstack([ex[0] for ex in examples]),
                        np.vstack([ex[1] for ex in examples]))
            batches = [(examples[0][i: i + batch_size], examples[1][i: i + batch_size])
                       for i in range(0, len(examples[-1]) + 1 - batch_size, batch_size)]

            if len(examples[-1]) % batch_size != 0:
                batches.append((np.vstack((examples[0][-(len(examples[-1]) % batch_size):],
                                           examples[0][:batch_size - (len(examples[0]) % batch_size)])),
                                np.vstack((examples[1][-(len(examples[-1]) % batch_size):],
                                           examples[1][:batch_size - (len(examples[-1]) % batch_size)]))))

            for batch in batches:  # mini batch
                # train_data(?, 7, 80, 2), train_label(?, 7)
                step += 1
                x_VAD = torch.from_numpy(batch[0]).to(DEVICE)
                y_VAD = torch.from_numpy(batch[1]).to(DEVICE)

                postnet_output = model(x_VAD)
                loss, postnet_loss = loss_fn(model, y_VAD, postnet_output)
                total_loss += loss.detach().item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
                optimizer.step()
                scheduler.step_update(step)
                lr = optimizer.param_groups[0]["lr"]

                total_loss_window.append(loss.detach().item())
                post_loss_window.append(postnet_loss.detach().item())

                postnet_accuracy = prediction(y_VAD, postnet_output)
                post_acc_window.append(float(postnet_accuracy))

                wandb.log({'loss': loss.item(), 'postnet_loss': postnet_loss.item(
                ), 'accuracy': postnet_accuracy, 'learning_rate': lr})

                if step % 10 == 0:
                    print('{}  E:{}, S: {}, overall loss: {:.4f}, '
                          'postnet loss: {:.4f}, post acc: {:.4f}, lr: {:.6f}'.format(
                              datetime.now().strftime(_format)[
                                  :-3], epoch, step,
                              total_loss_window.average, post_loss_window.average, post_acc_window.average, lr))

                if step % 10_000 == 0 and step > 400_000:
                    print('{} save checkpoint.'.format(
                        datetime.now().strftime(_format)[:-3]))
                    checkpoint = {
                        "model": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'step': step,
                        'scheduler': scheduler.state_dict()
                    }
                    if not os.path.isdir("./checkpoint"):
                        os.mkdir("./checkpoint")
                    torch.save(checkpoint, './checkpoint/weights_%s_%s.pth' %
                               (str(epoch), str(step / 1_000_000)))
                    torch.cuda.empty_cache()

            del batches, examples
            examples = []

    torch.cuda.empty_cache()
    return total_loss / (step - start_stpe + 1), step


def test_epoch(model, test_loader, loss_fn):
    print('{}  Validation Begins...'.format(
        datetime.now().strftime(_format)[:-3]))

    counter = 0
    total_loss = 0
    post_acc = 0

    for test_data, test_label in tqdm(test_loader):
        # test_data(?, 7, 80), train_label(?, 7)
        test_data = test_data[0]
        test_label = test_label[0]
        test_data = torch.as_tensor(
            test_data, dtype=torch.float32).to(DEVICE)
        test_label = torch.as_tensor(
            test_label, dtype=torch.float32).to(DEVICE)
        counter += 1
        postnet_output = model(test_data)
        postnet_accuracy = prediction(test_label, postnet_output)
        post_acc += float(postnet_accuracy)
        loss, _ = loss_fn(model, test_label, postnet_output)
        total_loss += loss.detach().item()

    # clear cache
    torch.cuda.empty_cache()
    print('{}  Validation Finished...'.format(
        datetime.now().strftime(_format)[:-3]))
    print('Loss: {:.4f}, post acc: {:.4f}'.format(
        total_loss / counter, post_acc / counter))
    return total_loss / counter, post_acc / counter


def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, batch_size, start_epoch=0,
          epochs=20, start_step=0):
    train_losses = []
    test_losses = []

    for e in range(start_epoch, epochs):
        train_loss, start_step = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, batch_size, e,
                                             start_step)
        train_losses.append(train_loss)

        with torch.inference_mode():
            test_loss, test_acc = test_epoch(model, test_loader, loss_fn)

        test_losses.append(test_loss)
        torch.cuda.empty_cache()
        print("{}  Epoch: {}/{}...".format(datetime.now().strftime(_format)[:-3], e + 1, epochs),
              "Train Loss: {:.5f}...".format(train_loss),
              "Test Loss: {:.5f}".format(test_loss))

        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": e + 1,
            'step': start_step,
            'scheduler': scheduler.state_dict()
        }

        wandb.log({'epoch': e + 1, 'train_epoch_loss': train_loss, 'valid_epoch_loss': test_loss,
                   'valid_epoch_accuracy': test_acc * 100})

        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(
            checkpoint, './checkpoint/weights_{}_acc_{:.2f}.pth'.format(e + 1, test_acc * 100))

    return train_losses, test_losses


def build_scheduler(optimizer, n_iter_step=1000, total_epoch=5., warmup=0.5):
    num_steps = int(total_epoch * n_iter_step)
    warmup_steps = int(warmup * n_iter_step)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=hparams.lr_min,
        warmup_lr_init=hparams.warmup_lr_init,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Tr-VAD Training Stage')
    parser.add_argument('--train_data_path', type=str, 
                        default='.../TIMIT_augmented/train_data/list.txt',
                        help='Path to the training metadata')
    parser.add_argument('--resume_train', 
                        action='store_true', help='whether continue training using checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/weights_10_acc_97.09.pth',
                        help='Path to the checkpoint file, if `resume_train` is set, then resume training from the checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    hparams = HParams()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = args.train_data_path
    print(f'training data_path: {input_path}')
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="Tr-VAD Training and Validation",

        # track hyperparameters and run metadata
        config={
            "learning_rate": hparams.lr,
            "architecture": "Tr-VAD",
            "dataset": "TIMIT x NOISEX-92",
            "epochs": hparams.epochs,
        }
    )

    RESUME = args.resume_train
    metadata_filename = args.train_data_path
    metadata, training_idx, validation_idx = train_valid_split(
        metadata_filename, test_size=0.05, seed=0)

    w, u = hparams.w, hparams.u
    winlen = hparams.winlen
    winstep = hparams.winstep
    fft = hparams.n_fft
    batch_size = hparams.batch_size

    train_dataset = SpeechDataset(metadata_filename, list(
        np.array(metadata)[training_idx]), hparams)
    valid_dataset = SpeechDataset(metadata_filename, list(
        np.array(metadata)[validation_idx]), hparams)

    train_loader = DataLoader(train_dataset, 1, True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(valid_dataset, 1, False,
                             num_workers=4, pin_memory=True)

    torch.cuda.empty_cache()
    model = VADModel(dim_in=hparams.dim_in, d_model=hparams.d_model, units_in=hparams.units_in,
                     units=hparams.units, layers=hparams.layers, P=hparams.P, drop_rate=hparams.drop_rate,
                     activation=hparams.activation).to(DEVICE)

    get_parameter_number(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    scheduler = build_scheduler(
        optimizer, n_iter_step=hparams.n_iter_step, total_epoch=hparams.epochs // 2, warmup=hparams.warmup_factor)

    wandb.watch(model, log="all")

    if not RESUME:
        print('{}  New Training...'.format(
            datetime.now().strftime(_format)[:-3]))

        train_losses, test_losses = train(model, train_loader, test_loader, add_loss, optimizer, scheduler, batch_size,
                                          epochs=hparams.epochs)
    else:
        print('{}  Resume Training...'.format(
            datetime.now().strftime(_format)[:-3]))
        path_checkpoint = args.checkpoint_path
        print('path_checkpoint: {}'.format(path_checkpoint))

        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        scheduler.load_state_dict(checkpoint['scheduler'])

        wandb.watch(model, log="all")

        train_losses, test_losses = train(model, train_loader, test_loader, add_loss, optimizer, scheduler,
                                          batch_size=batch_size, start_epoch=start_epoch, epochs=hparams.epochs, start_step=step)

    f = plt.figure()
    plt.grid()
    plt.plot(test_losses, label='valid')
    plt.plot(train_losses, label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.savefig('loss.png')
    plt.show()
    print("Min test loss: {:.6f}, min train loss: {:.6f}".format(
        min(test_losses), min(train_losses)))
