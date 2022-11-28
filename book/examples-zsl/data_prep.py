import pandas as pd
import pickle
import glob
import os


""" Word-Audio data preparation.
"""
def prepare_zsl_split_word_audio(): 

    # (1) Seen / unseen word classes are manually selected.
    # Note that, we keep the order of word classes the same as the order of audio classes.
    # (e.g. 'trumpet' -> 'Trumpet in C')
    
    seen_word_classes = [
        'bassoon', 
        'cello', 
        'flute', 
        'oboe', 
        'trumpet', 
        'tuba', 
        'violin'
    ]
    unseen_word_classes = [
        'accordion', 
        'clarinet', 
        'contrabass', 
        'horn', 
        'saxophone', 
        'trombone', 
        'viola'
    ]

    seen_audio_classes = [
        'Bassoon',
        'Cello',
        'Flute',
        'Oboe',
        'Trumpet in C',
        'Bass Tuba',
        'Violin',
    ]
    unseen_audio_classes = [
        'Accordion',
        'Clarinet in Bb',
        'Contrabass',
        'French Horn',
        'Alto Saxophone',
        'Trombone',
        'Viola'
    ]

    # (2) Get the audio split.
    (
        seen_audio_X_train, 
        seen_audio_y_train, 
        seen_audio_X_test, 
        seen_audio_y_test,
        unseen_audio_X_train, 
        unseen_audio_y_train, 
        unseen_audio_X_test, 
        unseen_audio_y_test,
    ) = _get_audio_seen_unseen_split(seen_audio_classes, unseen_audio_classes)

    # (3) Get the glove embeddings for each class word (prepared and saved beforehand).
    with open('./data/inst_glove_vector.p', 'rb') as f:
        inst_word_emb_dict = pickle.load(f)

    return (
        seen_word_classes,
        seen_audio_X_train, 
        seen_audio_y_train, 
        seen_audio_X_test, 
        seen_audio_y_test,
        unseen_word_classes,
        unseen_audio_X_train, 
        unseen_audio_y_train, 
        unseen_audio_X_test, 
        unseen_audio_y_test,
        inst_word_emb_dict
    )

""" Image-Audio data preparation.
"""
def prepare_zsl_split_img_audio():    

    # (1) Get the seen and unseen image classes and paths. 
    # Note that, we keep the order of image classes the same as the order of audio classes.
    # (e.g. 'trumpet' -> 'Trumpet in C')
    
    seen_img_classes = [
        'bassoon',
        'flute',
        'trumpet',
        'violin'
    ]
    unseen_img_classes = [
        'cello',
        'clarinet',
        'frenchhorn',
        'saxophone',
    ]

    seen_audio_classes = [
        'Bassoon',
        'Flute',
        'Trumpet in C',
        'Violin'
    ]
    unseen_audio_classes = [
        'Cello',
        'Clarinet in Bb',
        'French Horn',
        'Alto Saxophone',
    ]

    # Seen and unseen image classes are already split into subfolders.
    data_dir = './data/ppmip/seen' 
    seen_img_path = []
    seen_img_label = []
    for f in glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True):
        seen_img_path.append(os.path.relpath(f))
        seen_img_label.append(seen_img_classes.index(os.path.relpath(f).split('/')[-3]))

    data_dir = './data/ppmip/unseen'
    unseen_img_path = []
    unseen_img_label = []
    for f in glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True):
        unseen_img_path.append(os.path.relpath(f))
        unseen_img_label.append(unseen_img_classes.index(os.path.relpath(f).split('/')[-3]))

    # (2) Get the audio split.
    (
        seen_audio_X_train, 
        seen_audio_y_train, 
        seen_audio_X_test, 
        seen_audio_y_test,
        unseen_audio_X_train, 
        unseen_audio_y_train, 
        unseen_audio_X_test, 
        unseen_audio_y_test,
    ) = _get_audio_seen_unseen_split(seen_audio_classes, unseen_audio_classes)

    return (
        seen_img_classes,
        seen_img_path,
        seen_img_label,
        seen_audio_X_train, 
        seen_audio_y_train, 
        seen_audio_X_test, 
        seen_audio_y_test,
        unseen_img_classes,
        unseen_img_path,
        unseen_img_label,
        unseen_audio_X_train, 
        unseen_audio_y_train, 
        unseen_audio_X_test, 
        unseen_audio_y_test,
    )


""" Get the audio seen / unseen split.
"""
def _get_audio_seen_unseen_split(seen_audio_classes, unseen_audio_classes):
    ts_df = pd.read_csv('./data/TinySOL/TinySOL_metadata.csv')
    seen_audio_X_train, seen_audio_y_train, seen_audio_X_test, seen_audio_y_test = [], [], [], []
    for i in range(len(seen_audio_classes)):
        curr_class_text = seen_audio_classes[i]

        # We use the balanced train-test fold given by the dataset.
        curr_x_tr = ts_df.loc[(ts_df['Instrument (in full)'] == curr_class_text) & (ts_df['Fold'] < 4)]['Path'].tolist()
        curr_x_ts = ts_df.loc[(ts_df['Instrument (in full)'] == curr_class_text) & (ts_df['Fold'] == 4)]['Path'].tolist()
        curr_y_tr = [i] * len(curr_x_tr)
        curr_y_ts = [i] * len(curr_x_ts)
        seen_audio_X_train.extend(curr_x_tr)
        seen_audio_y_train.extend(curr_y_tr)
        seen_audio_X_test.extend(curr_x_ts)
        seen_audio_y_test.extend(curr_y_ts)

    # Same procedure for unseen classes.
    unseen_audio_X_train, unseen_audio_y_train, unseen_audio_X_test, unseen_audio_y_test = [], [], [], []
    for i in range(len(unseen_audio_classes)):
        curr_class_text = unseen_audio_classes[i]
        curr_x_tr = ts_df.loc[(ts_df['Instrument (in full)'] == curr_class_text) & (ts_df['Fold'] < 4)]['Path'].tolist()
        curr_x_ts = ts_df.loc[(ts_df['Instrument (in full)'] == curr_class_text) & (ts_df['Fold'] == 4)]['Path'].tolist()
        curr_y_tr = [i] * len(curr_x_tr)
        curr_y_ts = [i] * len(curr_x_ts)
        unseen_audio_X_train.extend(curr_x_tr)
        unseen_audio_y_train.extend(curr_y_tr)
        unseen_audio_X_test.extend(curr_x_ts)
        unseen_audio_y_test.extend(curr_y_ts)

    return (
        seen_audio_X_train, 
        seen_audio_y_train, 
        seen_audio_X_test, 
        seen_audio_y_test,
        unseen_audio_X_train, 
        unseen_audio_y_train, 
        unseen_audio_X_test, 
        unseen_audio_y_test,
    )
