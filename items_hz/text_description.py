import streamlit as st
from PIL import Image
import re
import unicodedata
import nltk
from cleantext import clean
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def text_description():
    
    st.markdown("<h3>After Translation</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Data cleaning approaches</h3>", unsafe_allow_html = True)

        st.write("1. The Unicode characters in the text were converted to ASCII. "
                 "2. The text data was transformed to lowercase. "
                 "3. URLs and email addresses were removed. "
                 "4. Special characters and punctuation were eliminated. "
                 "5. Repeated characters were removed. "
                 "6. Extra spaces, tabs, and new lines were cleared. "
                 "7. Each word was further stemmed and lemmatized. "
                 "8. Stop words were updated and removed from the remaining text. ")
        
        st.markdown("<h3>word cloud of each category</h3>", unsafe_allow_html = True)
        img = Image.open("images_hz/text wordcloud.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>top 10 most frequent words of each category</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.8725")
        img = Image.open("images_hz/word top10.jpg")
        st.image(img, use_container_width = True)


    stop_words = set(stopwords.words('english'))
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
