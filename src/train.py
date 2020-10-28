from model import EncodeDecodeModel
from data_loader import StagoDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from tqdm import tqdm
import torch


PATH_IMAGE_TRAIN = "../data/train/image"
PATH_AUDIO_TRAIN = "../data/train/audio"
PATH_IMAGE_VALID = "../data/train/image"
PATH_AUDIO_VALID = "../data/train/audio"
audio_fraction = 1.0
image_fraction = 1.0
num_step_on_training_epoch = 1000
num_step_on_valid_epoch = 500
EPOCHS = 100
LR = 0.0001
BATCH_SIZE = 4
DEVICE = 'cuda'


def train():
    stego_model = EncodeDecodeModel()
    train_dataset = StagoDataset(PATH_IMAGE_TRAIN, PATH_AUDIO_TRAIN,
                                 num_step_on_training_epoch, BATCH_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    valid_dataset = StagoDataset(PATH_IMAGE_VALID, PATH_AUDIO_VALID,
                                 num_step_on_valid_epoch, BATCH_SIZE)

    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True)

    loss_func1 = nn.SmoothL1Loss(reduction='mean')
    loss_func2 = nn.SmoothL1Loss(reduction='mean')
    print(stego_model)

    optimizer = Adam(stego_model.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    min_loss = 10e5
    
    for epoch_i in range(EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        total_loss = 0
        total_hidding_loss = 0
        total_revealing_loss = 0
        stego_model.to(DEVICE)
        stego_model.train()

        for step, batch in enumerate(tqdm(train_dataloader)):
            b_img_in = batch['image'].to(DEVICE)
            b_audio_in = batch['audio'].to(DEVICE)

            optimizer.zero_grad()
            img_out, audio_out = stego_model(b_img_in, b_audio_in)
            
            hidding_loss = loss_func1(b_img_in, img_out)
            revealing_loss = loss_func2(b_audio_in, audio_out)
            loss = hidding_loss + revealing_loss
            
            total_hidding_loss += hidding_loss.item()
            total_revealing_loss += revealing_loss.item()
            total_loss += image_fraction * hidding_loss.item()
            total_loss += audio_fraction * revealing_loss.item()

            loss.backward()
            optimizer.step()
        scheduler.step()

        avg_train_loss = total_loss / (num_step_on_training_epoch)
        avg_train_hidding_loss = total_hidding_loss / (num_step_on_training_epoch)
        avg_train_revealing_loss = total_revealing_loss / (num_step_on_training_epoch)
        print('Loss on train:')
        print(avg_train_loss)
        print(avg_train_hidding_loss)
        print(avg_train_revealing_loss)
        
        # Validation
        
        stego_model.eval()
        total_loss = 0
        total_hidding_loss = 0
        total_revealing_loss = 0

        for step, batch in enumerate(tqdm(valid_dataloader)):
            b_img_in = batch['image'].to(DEVICE)
            b_audio_in = batch['audio'].to(DEVICE)
            
            img_out, audio_out = stego_model(b_img_in, b_audio_in)
            
            hidding_loss = loss_func1(b_img_in, img_out)
            revealing_loss = loss_func2(b_audio_in, audio_out)
            loss = hidding_loss + revealing_loss
            
            total_hidding_loss += hidding_loss.item()
            total_revealing_loss += revealing_loss.item()
            total_loss += image_fraction * hidding_loss.item()
            total_loss += audio_fraction * revealing_loss.item()

        avg_valid_loss = total_loss / (num_step_on_valid_epoch)
        avg_valid_hidding_loss = total_hidding_loss / (num_step_on_valid_epoch)
        avg_valid_revealing_loss = total_revealing_loss / (num_step_on_valid_epoch)
        
        print('Loss on validation:')
        print(avg_valid_loss)
        print(avg_valid_hidding_loss)
        print(avg_valid_revealing_loss)
        
        # Save checkpoints
        
        if min_loss > avg_valid_loss:
            min_loss = avg_valid_loss
            torch.save(stego_model.encoder, '../pre-trained/encoder_{}.pt'.format(str(epoch_i)))
            torch.save(stego_model.decoder, '../pre-trained/decoder_{}.pt'.format(str(epoch_i)))
        
if __name__ == '__main__':
    train()
