import time
import requests
from bs4 import BeautifulSoup
import pandas as pd


def mp3getter(lst):  # Gets all the mp3 of the given languages
    url = "http://accent.gmu.edu/soundtracks/"
    for j in range(len(lst)):
        for i in range(1, lst[j][1]+1):
            while True:
                try:
                    fname = f"{lst[j][0]}{i}"
                    mp3 = requests.get(url+fname+".mp3")
                    print(f"\nDownloading {fname}.mp3")
                    with open(f"Audio/{fname}.mp3", "wb") as audio:
                        audio.write(mp3.content)
                except:
                    # Once file finishes downloading, a buffer time to make sure next download doesn't start too early
                    time.sleep(2)
                else:
                    break  # To break the while loop


def get_num(language):  # Returns the num of samples for a given language, useful in below function
    url = 'http://accent.gmu.edu/browse_language.php?function=find&language=' + language
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    div = soup.find_all('div', 'content')
    try:
        num = int(div[0].h5.string.split()[2])
    except AttributeError:
        num = 0
    return num


# Returns a list of tuples, (lang, num), mainly used for the mp3getter function
def get_formatted_languages(languages):
    formatted_languages = []
    for language in languages:
        num = get_num(language)
        if num != 0:
            formatted_languages.append((language, num))
    return formatted_languages


def get_speaker_info(start, stop):
    '''
    Inputs: two integers, corresponding to min and max speaker id number per language
    Outputs: Pandas Dataframe containing speaker filename, birthplace, native_language, age, sex, age_onset of English
    '''
    user_data = []
    for num in range(start, stop):
        info = {'speakerid': num, 'filename': 0, 'birthplace': 1,
                'native_language': 2, 'age': 3, 'sex': 4, 'age_onset': 5}
        url = "http://accent.gmu.edu/browse_language.php?function=detail&speakerid={}".format(
            num)
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'html.parser')
        body = soup.find_all('div', attrs={'class': 'content'})
        try:
            info['filename'] = str(body[0].find('h5').text.split()[0])
            bio_bar = soup.find_all('ul', attrs={'class': 'bio'})
            info['birthplace'] = str(bio_bar[0].find_all('li')[0].text)[13:-6]
            info['native_language'] = str(
                bio_bar[0].find_all('li')[1].text.split()[2])
            info['sex'] = str(bio_bar[0].find_all(
                'li')[3].text.split()[3].strip())
            user_data.append(info)
            info['']
        except:
            info['filename'] = ''
            info['birthplace'] = ''
            info['native_language'] = ''
            info['age'] = ''
            info['sex'] = ''
            info['age_onset'] = ''
            user_data.append(info)
        print(num)
        df = pd.DataFrame(user_data)
    df.to_csv('Data/speaker_info_all.csv')


# Extracting data of the required languages from the dataset
def extract_from_data(langs):
    # Dont execute the code in comments
    df1 = pd.read_csv("Data/speaker_info_2920-2940.csv")
    df = pd.read_csv("Data/speaker_info_all.csv")
    df = df[df['native_language'].isin(langs)]
    df1 = df1[df1['native_language'].isin(langs)]
    final = pd.concat([df, df1], axis=0)
    final.drop(['Unnamed: 0','age','age_onset'], axis = 1, inplace = True)
    final.to_csv('Data/final_data_2.csv')


if __name__ == "__main__":
    langs = ['arabic', 'hindi', 'spanish']
    lang_tuple = get_formatted_languages(langs)
    print(lang_tuple)
    print('Downloading now...')
    #mp3getter(lang_tuple)
    #get_speaker_info(1, 2942)
    extract_from_data(langs)
    print("DONE!!")
