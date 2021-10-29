import os
import pandas as pd

#path where data is stored
path_to_files = 'D:\emodb_dataset'

#function to read data
def read_data():
    #list  of emotion data
    emotion = []
    #list of paths of files in wav format
    path = []
    names = []
    for root, dirs, files in os.walk(path_to_files):
        for name in files:
            #identify male
            if name[0:2] in '0310111215':
                #Arger (Wut) -> Angry
                if name[5] == 'W':
                    emotion.append('male_angry')
                #Langeweile -> Boredom
                elif name[5] == 'L':
                    emotion.append('male_bored')
                # Ekel -> Disgusted
                elif name[5] == 'E':
                    emotion.append('male_disgust')
                #Angst -> Angry
                elif name[5] == 'A':
                    emotion.append('male_fear')
                #Freude -> Happiness
                elif name[5] == 'F':
                    emotion.append('male_happy')
                #Trauer -> Sadness
                elif name[5] == 'T':
                    emotion.append('male_sad')
                #Neutral
                elif name[5] == 'N':
                    emotion.append('male_neutral')
                else:
                    emotion.append('unknown')
            else:
                #identify female
                #Arger (Wut) -> Angry
                if name[5] == 'W':
                    emotion.append('female_angry')
                #Langeweile -> Boredom
                elif name[5] == 'L':
                    emotion.append('female_bored')
                # Ekel -> Disgusted
                elif name[5] == 'E':
                    emotion.append('female_disgust')
                #Angst -> Angry
                elif name[5] == 'A':
                    emotion.append('female_fear')
                #Freude -> Happiness
                elif name[5] == 'F':
                    emotion.append('female_happy')
                #Trauer -> Sadness
                elif name[5] == 'T':
                    emotion.append('female_sad')
                #Neutral
                elif name[5] == 'N':
                    emotion.append('female_neutral')
                else:
                    emotion.append('unknown')
            names.append(name)
            path.append(os.path.join(path_to_files,name))
        emodb_df = pd.DataFrame(emotion, columns=['labels'])
        emodb_df['source'] = 'EMODB'
        emodb_df = pd.concat([emodb_df, pd.DataFrame(names, columns=['names'])], axis=1)
        emodb_df = pd.concat([emodb_df, pd.DataFrame(path, columns=['paths'])], axis=1)
        
        return emodb_df

Data = read_data()
Data.to_csv('Emodb_data.csv')
                