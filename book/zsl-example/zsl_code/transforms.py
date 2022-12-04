from torchvision import transforms
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import (
    Compose,
    Lambda,
)

def get_transforms():
    img_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    SAMPLING_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = 256
    N_MELS = 80
    EPS = 1e-10
    mel_transform = Compose(
        [
            MelSpectrogram(
                sample_rate=SAMPLING_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
            ),
            Lambda(lambda x: x.clamp(min=EPS)),
            AmplitudeToDB(stype='power', top_db=80.),
            Lambda(lambda x: x / 80.),
            Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
        ]
    )

    return img_transforms, mel_transform

