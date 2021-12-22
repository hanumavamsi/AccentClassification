import pandas as pd
import os
import shutil

hindi_start = 35
start_num = 2943
hindi_new_data = []

three_langs = pd.read_csv('Data/three_languages.csv')
all_langs = pd.read_csv('Data/final_data.csv')

hindi_extra = os.listdir('WavFormat/andthis')
for file in hindi_extra:
    info = {'speakerid': start_num, 'filename': 1, 'birthplace': 2, 'native_language': 3, 'sex': 4}
    start_num += 1
    file_name = file.split('_')[0]
    new_file_name = f'hindi{hindi_start}'
    row_num = all_langs[all_langs['filename'] == file_name].index[0]
    info['filename'] = new_file_name
    hindi_start += 1
    info['birthplace'] = all_langs.loc[row_num].birthplace
    info['native_language'] = all_langs.loc[row_num].native_language
    info['sex'] = all_langs.loc[row_num].sex
    hindi_new_data.append(info)
    shutil.move(f'WavFormat/andthis/{file}',f'WavFormat/{new_file_name}.wav')

three_langs.pop('Unnamed: 0')
four_langs = three_langs.copy()
hindi_old = all_langs[all_langs['native_language'] == 'hindi']
hindi_old.pop('Unnamed: 0')
new_hindi = pd.DataFrame(hindi_new_data)

final_data = pd.concat([four_langs, hindi_old, new_hindi], axis=0)
final_data.to_csv('Data/four_languages.csv')