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
    

    st.markdown("<h3>Information about the text</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Word count and distribution of each categories</h3>", unsafe_allow_html = True)
        img = Image.open("images/text_1.jpg")
        st.image(img, use_container_width = True)
        st.write("""(a) representation of the 'prdtypecode' in the dataset. 
                 (b) word count of the 'designation' column. 
                 (c) word count of the 'description' column.""")
        
        img = Image.open("images/text_2.jpg")
        st.image(img, use_container_width = True)
        st.write("""(a,b) Histogram of detected languages in 'designation' and ‘description’. 
                 The majority of text is French followed by English. 
                 (c,d) Confidence of the detected language for French, English, German showing the multi-language character of input strings.""")

    st.markdown("<h3>Preprocessing Text</h3>", unsafe_allow_html = True)
    st.write("""1. Preprocessing before translation
             2. The translation
             3. Preprocessing after translation""")


    st.markdown("<h4>1. Preprocessing before translation</h4>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.write("""1. fragments of HTML markup language. e.g. tags like <br />Capacité de charge jusqu&#39;à 150 kg<br />
                 2. non utf-8 characters, e.g. characters encoded in cp1252/Windows-1252 end others
                 3. numerous characters that serve formatting, directional or layout purposes (invisible characters or non-printing characters), e.g. (\u200e, \u200b, \xad).
                 """)
        with st.expander("A comprehensive but probably not complete list is:"):
            st.write("""
                1. Directionality Marks
                \u200E: Left-to-Right Mark (LRM)
                9
                Helps ensure left-to-right direction in bidirectional text. \u200F: Right-to-Left Mark (RLM)
                Ensures right-to-left direction in bidirectional text. \u202A: Left-to-Right Embedding (LRE)
                Starts a left-to-right embedding, overriding the direction of surrounding text. \u202B: Right-to-Left Embedding (RLE)
                Starts a right-to-left embedding.
                \u202C: Pop Directional Formatting (PDF)
                Ends the effect of the last LRE or RLE.
                \u202D: Left-to-Right Override (LRO)
                Forces left-to-right direction for all characters until turned off. \u202E: Right-to-Left Override (RLO)
                Forces right-to-left direction.
                2. Zero Width Characters
                \u200B: Zero Width Space (ZWSP)
                Allows line breaks without visible spaces.
                \u200C: Zero Width Non-Joiner (ZWNJ)
                Prevents characters from being combined into a ligature or connected form. \u200D: Zero Width Joiner (ZWJ)
                Causes two characters to combine into a single glyph, useful in emojis and ligatures.
                3. Soft Hyphens and Other Hyphens \u00AD: Soft Hyphen (SHY)
                Appears as a hyphen only if a line break occurs at that position. \u2010: Hyphen
                A visible hyphen character.
                \u2011: Non-Breaking Hyphen
                Similar to a regular hyphen but prevents a line break. 4. Non-Breaking Spaces and Similar
                \u00A0: Non-Breaking Space (NBSP) Acts as a space but prevents line breaks.
                

                \u202F: Narrow Non-Breaking Space
                A narrower version of the NBSP, used in some languages like French. \u205F: Medium Mathematical Space
                A small amount of space, usually for mathematical text.
                \u3000: Ideographic Space
                Full-width space for East Asian text.
                5. Invisible Control Characters \u2060: Word Joiner (WJ)
                Prevents line breaks without adding visible space (similar to NBSP but with stricter control). \uFEFF: Zero Width No-Break Space (also Byte Order Mark, BOM)
                Indicates byte order at the start of a text file or serves as an invisible space elsewhere. \u2066: Left-to-Right Isolate (LRI)
                Starts a left-to-right isolate to separate directional text. \u2067: Right-to-Left Isolate (RLI)
                Starts a right-to-left isolate.
                \u2068: First Strong Isolate (FSI)
                Uses the direction of the first strong character. \u2069: Pop Directional Isolate (PDI)
                Ends the isolate effect of LRI, RLI, or FSI.""")
            
        st.write("""For the preprocessing we used:
                 1. beautiful soup (Richardson, 2024) is a package that can be used to remove HTML markup strings from an input text
                 2. ftfy (Alonso, 2024) and unicodedata.normalize is used to find non utf-8 characters ( from a different code page like cp1252 ) and replace them by the corresponding utf-8 character. This will make the input text more homogeneous.
                 3. unicodedata is used to find all non-printing characters and replace them with a single space character""")
        
        img = Image.open("images/text_3.jpg")
        st.image(img, use_container_width = True)
        st.write("""Tracing back the development of each preprocessing step. 
                 Upper row 'designation', lower row 'description': string length, word count, number of altered strings and the number of duplicates.""")


    st.markdown("<h4>2. The translation</h4>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.write("""
                 1. the source language needs to be detected
                 We used freely usable api of google translate (Google, 2024b) via the 
                 python api package deep-translator (Baccouri, 2024).
                 2. the presence of multi-language text strings
                 Our translation is based on the confidence of the detected languages from 
                 three different language detection packages: langdetect, lingua and LiteRT 
                 (Google, 2024; Danilk, 2024; Stahl, 2024)
                 """)
        img = Image.open("images/text_4.jpg")
        st.image(img, use_container_width = True)
        st.write("""The result of our translation procedure: 
                 (a,b) histogram of the detected languages before and after translation. 
                 (c,d) histogram of the confidence of the detected language (French, English and German).""")


    st.markdown("<h4>3. Preprocessing after translation</h4>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.write("""
                 1. The Unicode characters in the text were converted to ASCII. 
                 2. The text data was transformed to lowercase. 
                 3. URLs and email addresses were removed. 
                 4. Special characters and punctuation were eliminated. 
                 5. Repeated characters were removed. 
                 6. Extra spaces, tabs, and new lines were cleared. 
                 7. Each word was further stemmed and lemmatized. 
                 8. Stop words were updated and removed from the remaining text. 
                 """)

    st.markdown("<h3>Display of processed text data</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
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
        
    user_input_word = st.text_input("Input a sentense to clean: ", 'Merry Christmas!')
    output_st = clean_text(user_input_word)

    st.write('Here is the sentense after cleaning :\n', output_st)
