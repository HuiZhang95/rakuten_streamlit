import streamlit as st
from PIL import Image
import re
import unicodedata
import nltk
from cleantext import clean
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt_tab')

def text_description():
    
    st.markdown("<h3>After Translation</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Data cleaning approaches</h3>", unsafe_allow_html = True)

        st.write("1. The Unicode characters in the text were converted to ASCII. \n\n"
                 "2. The text data was transformed to lowercase. \n\n"
                 "3. URLs and email addresses were removed. \n\n"
                 "4. Special characters and punctuation were eliminated. \n\n"
                 "5. Repeated characters were removed. \n\n"
                 "6. Extra spaces, tabs, and new lines were cleared. \n\n"
                 "7. Each word was further stemmed and lemmatized. \n\n"
                 "8. Stop words were updated and removed from the remaining text. \n\n")
        
        st.markdown("<h3>word cloud of each category</h3>", unsafe_allow_html = True)
        img = Image.open("images_hz/text wordcloud.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>top 10 most frequent words of each category</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.8725")
        img = Image.open("images_hz/word top10.jpg")
        st.image(img, use_container_width = True)


    stop_words = {'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at',
    'be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",
    'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",
    'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if',
    'in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",
    'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out',
    'over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than',
    'that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too',
    'under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who',
    'whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself',
    'yourselves'}
    stop_words.update(['the', 'and','for', 'from', 'was', 'what', 'with', 'this', \
                    'that',  'don', 'pure', 'lot', 'are', 'who', 'more', 'will', 'tab', \
                    'each' , 'would', 'but', 'not','its','all','your', 'last','over', \
                        'are','you', 'can', 'above', 'his','she','ready', 'yes', \
                        'size'])
    stop_words.remove("no")
    stemmer = EnglishStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    def lemmatization(words) :
        output = []
        for string in words :
            lemma = wordnet_lemmatizer.lemmatize(string)
            if (lemma not in output) : output.append(lemma)
        return output
    
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
    def clean_text(text):
        text = clean(
            text = text,
            fix_unicode = True,
            to_ascii=True,
            lower = True,
            replace_with_url = ' ',
            replace_with_email = ' ',
            lang = 'en'
        )
        # Remove all special characters and punctuation
        text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
        # Remove repeated characters
        text = re.sub(r'(.)\1{3,}',r'\1', text)
        # remove extra spaces, tabs, and new lines
        text = " ".join(text.split())
        w = str(text)
        w = unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z0-9?.!]+", " ", w)
        w = re.sub(r'\b\w{0,2}\b', '', w)
        # stem the word
        words = word_tokenize(w.strip())
        words2 = [stemmer.stem(word) for word in words]
        # lemmatize the word
        words3 = lemmatization(words2)

        # remove stopword
        words = [word for word in words3 if word not in stop_words]

        if len(words) < 1: # sometimes, all words are removed
            return w
        else:
            return ' '.join(words).strip()
        
    user_input_word = st.text_input("Input a sentense: ", 'Merry Christmas!')
    output_st = clean_text(user_input_word)

    st.write('Here is the sentense after cleaning :\n', output_st)
