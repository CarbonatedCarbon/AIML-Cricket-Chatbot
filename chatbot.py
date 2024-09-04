import os
import aiml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from nltk.sem import logic
from nltk.inference.resolution import ResolutionProverCommand, ResolutionProver
import torch
import time
time.clock = time.time
import requests
#tts and audio libraries
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import contextlib
from threading import Thread
#date lib
from datetime import datetime
import dateutil.parser
#translation libraries
import translators as ts
#classification stuff
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import keras

#constants
from constants import *

#fuzzy
import simpful as sf

#tensorflow settings

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')

#tts config
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the AIML kernel
kernel = aiml.Kernel()
kernel.verbose(False)
# Load the AIML files
kernel.learn("my_logic.xml")

# Load KB
knowledge_base = []
with open("cricket questions.txt", "r") as file:
    for line in file:
        if ":" in line:
            question, answer = line.strip().split(":", 1)
            knowledge_base.append((question.strip(), answer.strip()))

premises = []
# load FOL
with open('FOL_statements.txt', 'r') as csvfile:
    reader = csvfile.readlines()

    for row in reader:
        statement = logic.Expression.fromstring(row)
        premises.append(statement)
def play_speech(text, lang = "en"):
    """
    ENGLISH MODELS
    """
    # tts model #tts_models/en/ljspeech/vits--neon
    # #tts_models/en/blizzard2013/capacitron-t2-c150_v2
    # #tts_models/en/ljspeech/glow-tts
    """
    SPANISH MODEL
    """
    #tts_models / es / mai / tacotron2 - DDC
    """
    FRENCH MODEL
    """
    #tts_models/fr/mai/tacotron2-DDC
    """
    GERMAN MODEL
    """
    #tts_models/de/thorsten/tacotron2-DDC
    with contextlib.redirect_stdout(None):
        #start tts model
        if lang == 'en':
            tts = TTS(model_name="tts_models/en/ljspeech/vits--neon", progress_bar=False).to(device)
        elif lang == 'es':
            tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device)
        elif lang == 'fr':
            tts = TTS(model_name="tts_models/fr/mai/tacotron2-DDC", progress_bar=False).to(device)
        else:
            tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

        tts.tts_to_file(text=text, file_path="output.wav", service="coqui", vocoder_name="ljspeech/hifigan_v2",stream=False)

    #play the audio
    play(AudioSegment.from_wav("output.wav"))
#main loop TTS function
def play_audio(resp):
    global chosen_lang
    if resp == None:
        trans = "internal error"
    else:
        trans = ts.translate_text(resp, to_language=chosen_lang)

    # diplay response
    print("Bot:", trans)
    # play tts
    tts_audio = Thread(target=play_speech(trans, chosen_lang))
    tts_audio.start()
    #tts_audio.join()

def find_closest_answer(user_input):
    # Preprocess knowledge base for similarity calculations
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    knowledge_base_processed = [(question.lower(), answer.lower()) for question, answer in knowledge_base]
    knowledge_base_lemmatized = [(lemmatizer.lemmatize(question), lemmatizer.lemmatize(answer)) for question, answer in
                                 knowledge_base_processed]

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [f"{question} {answer}" for question, answer in knowledge_base_lemmatized])
    """
    vector of word frequencies
    """
    # Preprocess user input for similarity calculations
    knowledge_base_tokens = []
    for question, answer in knowledge_base_processed:
        # preprocess KB into tokens
        tokenized_question = nltk.word_tokenize(question)
        lemmatized_question = [lemmatizer.lemmatize(word.lower()) for word in tokenized_question]

        knowledge_base_tokens.append(lemmatized_question)

    # preprocess into tokens
    user_tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)]

    # Check for exact matches first
    for i, tokens in enumerate(knowledge_base_tokens):
        if user_tokens == tokens:
            return knowledge_base[i][1]

    # Use tokens for vectorization
    user_input_vector = tfidf_vectorizer.transform([' '.join(user_tokens)])
    knowledge_base_vectors = tfidf_vectorizer.transform([' '.join(tokens) for tokens in knowledge_base_tokens])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_input_vector, knowledge_base_vectors)

    """
    compares frequency of words in input and KB in form of
    """

    # find most similar score
    closest_match_index = similarity_scores.argmax()
    # print("closest == ", knowledge_base[closest_match_index][1])
    # print("\n",similarity_scores[0][closest_match_index])
    # Lower the similarity threshold
    similarity_threshold = 0.8

    if similarity_scores[0][closest_match_index] < similarity_threshold:
        return "I'm sorry, but I couldn't find a relevant answer to your question."

    return knowledge_base[closest_match_index][1]
def get_response(res):
    global premises

    #process user input
    #doc = nlp(user_input)



    # Use AIML for predefined responses
    aiml_response = kernel.respond(res)

    if aiml_response:
        # check if language option
        if aiml_response[0:3] == "#1$":
            global chosen_lang
            # get lang in response
            words = aiml_response.split(" ")
            response = get_country_code(words[-1])
            chosen_lang = response[0]
            #print(chosen_lang)
            if response[1] == 0:
                return aiml_response[3::]
            else:
                return "sorry, could not find the requested language"
        # "know" statement
        elif aiml_response[0:3] == "#4$":
            params = aiml_response.split('$')
            data, type = params[1].split(' IS ')
            check = False
            # check if type in KB
            for obj in premises:
                objs = str(obj)
                if type in objs:
                    check = True
                    break

            if check == False:

                return f"{type} is not a valid type"

            statement = logic.Expression.fromstring(f'{type}({data})')
            neg_statement = logic.Expression.fromstring(f'-{type}({data})')

            if statement in premises:
                return f" i already know {data} is a {type}"

            # check if conclusion is correct
            # FOR CHECK THAT, is_correct = ResolutionProverCommand(conclusion, premises).prove()




            added = premises + [statement]
            #check contradiction if added
            contradict = ResolutionProver().prove(goal=None, assumptions=added, verbose=False)#ResolutionProver().prove(goal=None, assumptions=added, verbose=True)
            #ResolutionProverCommand(None, added)
            #is_consistent = ResolutionProver().prove(statement, premises, verbose=True)
            #check neg
            #check_neg = logic.Expression.fromstring(f'-{type}({data})')

            #neg = ResolutionProverCommand(None, added)
            #neg = neg.prove()

            if contradict == False:


                premises.append(statement)

                return f"i will remember that {data} is {type}"
            else:

                return f"there is a contradiction in your statement, {data} cannot be {type}"
        # "check" statement
        elif aiml_response[0:3] == "#5$":
            params = aiml_response.split('$')
            data, type = params[1].split(' IS ')
            check = False
            # check if type in KB
            for obj in premises:
                objs = str(obj)
                if type in objs:
                    check = True

            if check == False:

                return f"{type} is not a valid type"



            statement = logic.Expression.fromstring(f'{type}({data})')
            neg_statement = logic.Expression.fromstring(f'-{type}({data})')

            is_correct = ResolutionProver().prove(goal=statement, assumptions=premises,verbose=False)
            if is_correct:

                return f"correct, {data} is {type}"
            else:

                #negate the statement for the resolution goal
                goal = logic.Expression.fromstring(f'-{type}({data})')

                contradict = ResolutionProverCommand(goal, premises)
                if contradict.prove(verbose=False):

                    return f"incorrect, {data} is not {type}"
                else:

                    return "i am not sure about that"
        # cricket api
        elif aiml_response[0:3] == "#2$":
            # read response
            # check statement
            type = aiml_response.split(" ")[1]
            if type == "series":
                play_audio("enter series cricket format or name:")
                format = input("You: ")
                play_audio(aiml_response[3::])
                # print list of available series
                # provide series info


                return cricket_summary_data(format,type)
            return "finished cricket tings"
        # player info
        elif aiml_response[0:3] == "#3$":
            # read response
            # check statement
            type = aiml_response.split(" ")[1]
            if type == "player":
                play_audio("enter players name:")
                format = input("You: ")
                play_audio(aiml_response[3::])
                # print list of available series
                # provide series info
                return cricket_summary_data(format, type)
        elif aiml_response[0:3] == "#7$":
            play_audio(aiml_response[3::])
            pic_path = select_file()

            answer = guess_picture(pic_path)

            return answer
        elif aiml_response[0:3] == "#6$":
            play_audio(aiml_response[3::])
            #gather variable data required from user
            m, g, com, c, s = get_pitch_variables()

            condition = pitch_conditions(m,g,com,c,s)

            return condition

        return aiml_response

    return find_closest_answer(user_input)
def get_cricket_data(option,format = ""):
    url = ""
    if option == "current":
        url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
    elif option == "series":
        url = f"https://api.cricapi.com/v1/series?apikey={api_key}&offset=0&search={format}"
    elif option == "players":
        url = f"https://api.cricapi.com/v1/players?apikey={api_key}&offset=0&search={format}"
    response = requests.get(url)
    matches = response.json()

    # Assuming the JSON response contains a list of matches under a 'data' key
    return matches
def list_cricket_names(matches):
    out_string = ""
    num = 0
    for match in matches.get("data", []):
        out_string += str(num) + ": " + match.get("name") + "\n"
        num +=1
    if num <= 1:
        return ""
    play_audio("please select from the following:")
    return out_string
def get_options(json):
    names = [tings.get("name") for tings in json.get("data", [])]

    if len(json.get("data",[])) == 1:
        return names[0]

    choice = input("You: select number (0-"+str(len(json.get("data",[]))-1)+"): ")

    return names[int(choice)]
# check if match date in future or past
def check_date(startDate):
    current = datetime.today().date()
    matchDate = dateutil.parser.parse(startDate).date()

    return matchDate > current #true if match newer than current
def clean_string(bad_string):
    string = filter(lambda x: x.isalpha() or x.isspace(), bad_string)
    return "".join(string)
def get_player_information(id):
    api_key = "293f1e12-8a9e-4c86-b8c7-65236c98ad07"

    url = f"https://api.cricapi.com/v1/players_info?apikey={api_key}&id={id}"
    response = requests.get(url)
    matches = response.json()

    # Assuming the JSON response contains a list of matches under a 'data' key
    return matches

def cricket_summary_data(format,type = ""):
    # filter by match format
    if type == "series":
        matches = get_cricket_data("series", format)
        out_string = list_cricket_names(matches)
        print(out_string)
        if out_string == None or out_string == "":
            return "could not find the requested format"
        # find match in matches
        choice = get_options(matches)
        out_string = ""
        for match in matches.get("data",[]):
            if match.get("name") == choice:
                # removes numbers and special characters
                title = clean_string(str(match.get("name")))
                #check if match date is in future
                if check_date(match.get("startDate")):
                    out_string += "the "+title+"will start on "+str(match.get("startDate"))+" and finish on "+match.get("endDate")+"\n"
                    out_string += "It will include a total of " + str(match.get("matches")) + " matches"
                    return out_string
                else:
                    out_string += "the "+title+"started on "+str(match.get("startDate"))+" and finished on "+match.get("endDate")+"\n"
                    out_string += "It included a total of "+str(match.get("matches"))+" matches"
                    return out_string
    elif type == "player":
        info = get_cricket_data("players", format)
        out_string = list_cricket_names(info)
        if out_string != "":
            print(out_string)
        choice = get_options(info)
        out_string = ""
        if out_string == None or out_string == None:
            return "could not find the requested player"
        # find match in matches
        for _ in info.get("data", []):
            if _.get("name") == choice:
                information = get_player_information(_.get("id"))
                stuff = information.get("data")
                if stuff is not None:
                    if stuff.get("battingStyle") != None and stuff.get("role") != None and stuff.get("bowlingStyle") != None and stuff.get("placeOfBirth") != None:

                        out_string += stuff.get("name") + " is a "+stuff.get("role")+" born in "+stuff.get("placeOfBirth")+"\n"
                        out_string += "they have a "+stuff.get("battingStyle")+" batting style and a "+stuff.get("bowlingStyle")+" bowling style"+"\n"
                    else:
                        out_string = f"no information about {choice}"


                return out_string


    return f"the requested {type} could not be found."
def get_country_code(name):
    lang_dict = {
        "spanish": 'es',
        "english": 'en',
        "french": 'fr',
        "german": 'de'
    }
    if name in lang_dict:
        return [lang_dict[name],0]
    else:
        return [chosen_lang,1]

#classification functions
def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [32, 32])
    img = tf.expand_dims(img, 0)
    img = img / 255.0
    return img
def select_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("All files", "*.*")])
    root.destroy()

    if file_path:
        return file_path
def guess_picture(path):
    model = keras.models.load_model('training code (submission)/saved_model')

    labels = ["drive", "legglance-flick", "pullshot", "sweep"]
    predict = model.predict(process_path(path))
    index = tf.argmax(predict, axis=1).numpy()[0]
    return f"the model predicted the image is the {labels[index]} batting action"
#fuzzy logic
def pitch_conditions(moist,grass,compact,clay,silt):
    with contextlib.redirect_stdout(None):
        FS = sf.FuzzySystem()

    #linguistic variables
    moisture_lv = sf.AutoTriangle(3, terms=['low', 'medium', 'high'], universe_of_discourse=[0, 100])
    grass_lv = sf.AutoTriangle(3, terms=['sparse', 'average', 'lush'], universe_of_discourse=[0, 100])
    compaction_lv = sf.AutoTriangle(3, terms=['low', 'medium', 'high'], universe_of_discourse=[0, 100])
    clay_content_lv = sf.AutoTriangle(3, terms=['low', 'medium', 'high'], universe_of_discourse=[0, 100])
    silt_sand_lv = sf.AutoTriangle(3, terms=['low', 'medium', 'high'], universe_of_discourse=[0, 100])

    #add to fuzzy system
    FS.add_linguistic_variable("moisture", moisture_lv)
    FS.add_linguistic_variable("grass", grass_lv)
    FS.add_linguistic_variable("compaction", compaction_lv)
    FS.add_linguistic_variable("clay_content", clay_content_lv)
    FS.add_linguistic_variable("silt_sand", silt_sand_lv)

    #fuzzy sets
    O1 = sf.TriangleFuzzySet(0, 0, 5, term="bowler")
    O2 = sf.TriangleFuzzySet(3, 5, 7, term="balanced")
    O3 = sf.TriangleFuzzySet(5, 10, 10, term="batsman")
    FS.add_linguistic_variable("pitch_condition", sf.LinguisticVariable([O1,O2,O3],universe_of_discourse=[0,10]))

    FS.add_rules([
        "IF (moisture IS high) AND (grass IS lush) AND (compaction IS low) THEN (pitch_condition IS bowler)",
        "IF (moisture IS high) AND (grass IS average) AND (compaction IS low) THEN (pitch_condition IS bowler)",
        "IF (silt_sand IS high) AND (compaction IS low) AND (moisture IS medium) THEN (pitch_condition IS batsman)",
        "IF (grass IS sparse) AND (moisture IS low) AND (compaction IS high) THEN (pitch_condition IS batsman)",
        "IF (clay_content IS high) AND (moisture IS low) THEN (pitch_condition IS balanced)",
        "IF (moisture IS high) AND (compaction IS high) THEN (pitch_condition IS batsman)"
    ])

    #assign values
    FS.set_variable("moisture", moist)  # Example value
    FS.set_variable("grass", grass)  # Example value
    FS.set_variable("compaction", compact)  # Example value
    FS.set_variable("clay_content", clay)  # Example value
    FS.set_variable("silt_sand", silt)  # Example value

    condition = FS.inference(verbose=False)

    if condition == "bowler":
        return "pitch is ideal for pace bowlers due to the predictable bounce and minimal pace absorption"
    elif condition == "balanced":
        return "the pitch favours neither batter or bowler"
    else:
        return "the pitch favours the batsman, as the ball would bounce slower and higher"
def get_pitch_variables():
    moist, grass, compact, clay, silt = "","","","",""
    #factors have been gathered from the website https://www.scienceabc.com/sports/whats-the-impact-of-pitch-on-the-game-of-cricket.html

    play_audio("how damp/wet is the pitch (0%-100%)");moist = input("You: ")

    play_audio("how grass-covered is the pitch (0%-100%)");grass = input("You: ")

    play_audio("has the pitch been rolled on? how kept is the compaction of the pitch (0%-100%)");compact = input("You: ")

    play_audio("how much of the pitch is covered in clay (0%-100%)");clay = input("You: ")

    play_audio("how much of the pitch is covered in silt (0%-100%)");silt = input("You: ")

    return int(moist), int(grass), int(compact), int(clay), int(silt)

#caches the translation stuff
#_ = ts.preaccelerate_and_speedtest()
global chosen_lang
chosen_lang = 'en'


# Main loop to interact with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    audio_thread = Thread(target=play_audio, args=(response,))
    audio_thread.start()
    audio_thread.join()
