import numpy as np
import librosa
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# This file can be used for noth 7 classes and 4 classes you need to keep in mind that 
# file will be edited for 4 classes by using the code givvenn below
emodb_data = pd.read_csv('/content/drive/MyDrive/Emodb_drive.csv')
emodb_data.head()

# for 4 classes
# below array is equivalent to [0,1,2,3,4,5,6] as in emodb_data dataframe
# labels = ['happy', 'neutral', 'angry', 'sad', 'fear', 'bored', 'disgust']
# emodb_data.drop(emodb_data.index[emodb_data['labels'] == 4], inplace=True)
# emodb_data.drop(emodb_data.index[emodb_data['labels'] == 5], inplace=True)
# emodb_data.drop(emodb_data.index[emodb_data['labels'] == 6], inplace=True)

def get_data(flatten=False, feature=1, mfcc_len=39, mslen = 35000, n_fft=512, hop_length=128):
    """
    Read the files get the data perform the test-train split and return them to the caller
    :param mfcc_len: Number of mfcc features to take for each frame
    :param flatten: Boolean specifying whether to flatten the data or not
    :return: 4 arrays, x_train x_test y_train y_test
    """
    # store features in here
    data = []

    # obtain labels directly from data frame
    labels = np.array(emodb_data.labels)
    fea=[]
    max_fs = 0
    min_sample = int('9' * 10)
    cnt = 0
    # for paths in dataframe's column paths
    for path in emodb_data.paths:
        # load the audio
        signal,fs = librosa.core.load(path,sr=16000)
        max_fs = max(max_fs, fs)
        s_len = len(signal)
        # pad the signals to have same size if lesser than required
        if s_len < mslen:
          pad_len = int(mslen - s_len)
          pad_rem = int(pad_len % 2)
          pad_len /= 2
          signal = np.pad(signal, (int(pad_len), int(pad_len + pad_rem)), 'constant', constant_values=0)
        # else slice them
        else:
          pad_len = int(s_len - mslen)
          pad_rem = pad_len % 2
          pad_len /= 2
          signal = signal[int(pad_len):int(pad_len + mslen)]
          min_sample = min(len(signal), min_sample)
        
        # Extract Mel Spectrogram features
        if feature==1:
          melspecfea = librosa.feature.melspectrogram(y=signal, sr=fs,n_fft=n_fft, hop_length=hop_length)
          melspecfea = librosa.power_to_db(melspecfea,ref=np.max)
          fea = (melspecfea- np.min(melspecfea))/np.ptp(melspecfea)
        
        # Extract Mfcc, delta, delta-delta features
        if feature==2:
          melspecfea = librosa.feature.melspectrogram(y=signal, sr=fs,n_fft=n_fft, hop_length=hop_length)
          mfcc = librosa.feature.mfcc(S=melspecfea, n_mfcc=20)
          mfcc_delta = librosa.feature.delta(mfcc)
          mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
          mfcc_all= np.vstack((mfcc,mfcc_delta,mfcc_delta2))
          fea = (mfcc_all- np.min(mfcc_all))/np.ptp(mfcc_all)
        
        # Extract Spectrogram features    
        if feature==3:
          spectrogram1 = librosa.core.stft(signal, n_fft=512)
          spectrogram1 = np.abs(spectrogram1)
          fea = (spectrogram1- np.min(spectrogram1))/np.ptp(spectrogram1)
      
        if flatten:
          # Flatten the data
          fea = fea.flatten()
        data.append(fea)
        cnt += 1
    # train test split on data  
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

