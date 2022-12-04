import time
import copy
import os
import datetime
import torch
from collections import OrderedDict
from tqdm import tqdm 

class ModelManager():
    def __init__(
        self,
        model, 
        loss, 
        optimizer, 
        scheduler, 
        device,
        task='word'
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task = task
        
    def train_model(self, 
        dataloaders, 
        exp, 
        validation='random', 
        num_epochs=200
    ):
        since = time.time()
        dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'test']}
        best_model_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    running_loss = 0.0
                    for inputs in tqdm(dataloaders[phase]):
                        inp_audio = inputs[0]
                        inp_pos= inputs[1]
                        inp_neg = inputs[2]

                        inp_audio = inp_audio.to(self.device)
                        inp_pos = inp_pos.to(self.device)
                        inp_neg = inp_neg.to(self.device)
                        self.optimizer.zero_grad()
                        
                        _a, _p, _n = self.model(
                            inp_audio,
                            inp_pos,
                            inp_neg
                        )
                        loss = self.loss(_a, _p, _n)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item() * inp_audio.size(0)

                    epoch_loss = running_loss / (dataset_sizes[phase] * dataloaders[phase].batch_size)
                    print(f'{phase} Loss: {epoch_loss:.4f}') # Acc: {epoch_acc:.4f}')

                elif phase == 'test':
                    if validation == 'random':
                        _, _, _, valid_loss = self.infer_random(dataloaders, phase)
                    elif validation == 'full':
                        _, _, _, valid_loss = self.infer_all(dataloaders, phase)
                    self.scheduler.step(valid_loss, epoch)
                
                if epoch != 0 and epoch % 5 == 0 and phase == 'train':
                    self.save_model(exp, epoch, epoch_loss)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def save_model(self, exp, epoch, epoch_loss):
        _d = datetime.datetime.now()
        curr_date = _d.strftime("%x").split('/')[2] + _d.strftime("%x").split('/')[0] + _d.strftime("%x").split('/')[1]
        curr_weight_path =  os.path.join(
            'checkpoints',
            '_'.join([exp, curr_date]),
            'w_ep-%05d_l-%.4f' % (epoch, epoch_loss)
        ) + '.pth'
        if not os.path.exists(os.path.dirname(curr_weight_path)):
            os.makedirs(os.path.dirname(curr_weight_path))
        torch.save(self.model.state_dict(), curr_weight_path)
        print('Model saved at :', curr_weight_path)


    def load_model(self, weight_path):
        state_dict = torch.load(weight_path, map_location=self.device)
        if torch.cuda.device_count() > 1:
            self.model.load_state_dict(state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        print('Loading model done:', weight_path)


    def infer_all(self, dataloaders, phase='test'):
        _dataset = dataloaders[phase].dataset
        self.model.eval()   # Set model to evaluate mode
        audio_embeddings, media_embeddings, labels, batch_lens = [], [], [], []
        running_loss = 0.0
        with torch.no_grad():
            for i in tqdm(range(len(_dataset.audio_paths))):
                _path = _dataset.audio_paths[i]
                _label = _dataset.audio_labels[i]
                wav, sr = _dataset._load_item(os.path.join(_dataset.audio_dir, _path))
                wav = _dataset._resample_item(wav, sr)
                wavs = _dataset._get_sequential_wav_seg_list(wav)
                audios = [_dataset.audio_transform(wav).unsqueeze(0) for wav in wavs]
                inp_audio = torch.stack(audios)
                inp_audio = inp_audio.to(self.device)

                batch_lens.append(inp_audio.size(0))

                inp_pos, inp_neg = _dataset._get_pos_neg_inp(i)
                inp_pos = torch.stack([inp_pos] * inp_audio.size(0)).to(self.device)
                inp_neg = torch.stack([inp_neg] * inp_audio.size(0)).to(self.device)

                _a, _p, _n = self.model(
                    inp_audio,
                    inp_pos,
                    inp_neg
                )
                
                audio_embeddings.append(torch.mean(_a, dim=0, keepdim=False)) # averaging full sequence to one embedding
                media_embeddings.append(torch.mean(_p, dim=0, keepdim=False)) # averaging full sequence to one embedding
                
                labels.append(_label)
                loss = self.loss(_a, _p, _n)
                running_loss += loss.item() * inp_audio.size(0)
            
        epoch_loss = running_loss / sum(batch_lens)
        
        print(f'{phase} Loss: {epoch_loss:.4f}') 
        audio_embeddings = torch.stack(audio_embeddings)
        media_embeddings = torch.stack(media_embeddings)
        
        return (
            audio_embeddings.detach().cpu().numpy(), 
            media_embeddings.detach().cpu().numpy(), 
            labels,
            epoch_loss
        )
        
    def infer_random(self, dataloaders, phase='test'): 
        dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'test']}
        self.model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        audio_embeddings, media_embeddings, labels = [], [], []
        with torch.no_grad():
            for inputs in dataloaders[phase]:
                inp_audio = inputs[0]
                inp_pos= inputs[1]
                inp_neg = inputs[2]
                label = inputs[3]

                inp_audio = inp_audio.to(self.device)
                inp_pos = inp_pos.to(self.device)
                inp_neg = inp_neg.to(self.device)
                self.optimizer.zero_grad()
                
                _a, _p, _n = self.model(
                    inp_audio,
                    inp_pos,
                    inp_neg
                )
                loss = self.loss(_a, _p, _n)
            

                audio_embeddings.append(_a)
                media_embeddings.append(_p)
                labels.extend(label.tolist())
                
                loss = self.loss(_a, _p, _n)
                running_loss += loss.item() * inp_audio.size(0)

        epoch_loss = running_loss / (dataset_sizes[phase] * dataloaders[phase].batch_size)
        print(f'{phase} Loss: {epoch_loss:.4f}') 
        audio_embeddings = torch.concat(audio_embeddings, 0)
        media_embeddings = torch.concat(media_embeddings, 0)
        
        return (
            audio_embeddings.detach().cpu().numpy(), 
            media_embeddings.detach().cpu().numpy(), 
            labels,
            epoch_loss
        )
