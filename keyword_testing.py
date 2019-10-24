import spacy
import pandas as pd
import gzip

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")
THRESH = 0.5
# 0.4 is too general
# 0.5 should work in b8ta recordings
# 0.6 works well on tech-specific audio

# For Testing
general_keywords = ['technology', 'computer', 'laptop', 'tablet', 'device', 'charging', 'smart', 'headphones', 'micro', 'discount', 'mobile', 'electric']
performance_keywords = ['speed', 'fast', 'display', 'size', 'brightness', 'battery', 'quality', 'sleek', 'lightweight', 'wireless', 'camera']


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def _read_file(filename, low=True):
    f = open(filename, 'r')
    text = f.read()
    if low:
        return text.lower()
    return text


def remove_stopwords_and_punctuation(text):
    """
    Remove stopwords, punctuation and special symbols from text and return lemmatized version of words

    :param text (string) String containing ASR data
    :return: filtered_text (list) List of relevant words
    """
    doc = nlp(text)
    extra_stopwords = ["hello", "yes", "left", "right", "like", "think", "want", "need", "cool", "okay", ""]
    for word in extra_stopwords:
        nlp.vocab[word].is_stop = True
    print(nlp.vocab["want"].is_stop)
    filtered_text = []
    for word in doc:
        if not word.is_stop and not word.is_punct and word.pos_ != 'NUM' and word.pos_ != 'SYM':
            filtered_text.append(str(word.lemma_))
    return filtered_text


def text_to_dict(text):
    """
    Create a dictionary of word frequencies from given list of words

    :param text (list)  List of words whose frequencies need to be calculated
    :return final_dict (dictionary)  Dictionary of word frequencies with word as key and frequency as values (value > 1 and length of key > 3)
    """
    data_dict = dict()
    final_dict = dict()

    for word in text:
        if word in data_dict:
            data_dict[word] += 1
        else:
            data_dict[word] = 1

    for key in data_dict.keys():
        if len(key) > 3 and data_dict[key] > 1:
            final_dict[key] = data_dict[key]

    return final_dict


def similar_words(data_dict, keywords):
    """
    Computes similarity between keys of data_dict and elements of keywords.
    Returns top matches from ASR transcript to keywords.

    :param data_dict: (dictionary) (key, value) => (word, frequency)
    :param keywords: (list) Contains list of domain and business keywords
    :return: final_dict (dictionary) (key, value) => (final keywords, frequency)
    """
    keyword_text = " ".join(keywords)
    key_tokens = nlp(keyword_text)

    text = " ".join(data_dict.keys())
    tokens = nlp(text)

    final_dict = dict()

    for token in tokens:
        for key_token in key_tokens:
            if token.has_vector and key_token.has_vector:
                if token.similarity(key_token) > THRESH:
                    final_dict[str(token.text)] = (data_dict[str(token.text)], token.similarity(key_token), key_token.text)
                    continue

    return final_dict


def keyword_generator(transcript, domain_keywords, client_keywords=[]):
    """
    Wrapper function to extract keywords (domain specific + client specific) form ASR transcript

    :param transcript: (string) Text from which keywords need to be extracted
    :param domain_keywords: (list) Necessary seed-keywords
    :param client_keywords: (list) Client specific keywords, Default: [], empty list
    :return: keyword_dict: (dictionary) (key, value) => (keywords extracted from transcript, (frequency of keywords, similarity value, source keyword)
    """
    clean_transcript = remove_stopwords_and_punctuation(transcript)
    keywords = domain_keywords + client_keywords
    keyword_dict = similar_words(text_to_dict(clean_transcript), keywords)
    return keyword_dict


def testing_function(path):
    df = get_df(path)
    reviews = list(df['reviewText'])
    fin_reviews = []
    cols = ['Candidate Keyword', 'Similarity Keyword', 'Source Keyword']
    keyword_df = pd.DataFrame(columns=cols)
    final_keys = []

    for review in reviews:
        if len(review.split()) > 40:
            fin_reviews.append(review)

    for review in fin_reviews[:100]:
        data_dict = (keyword_generator(review, performance_keywords, general_keywords))
        for key in data_dict.keys():
            final_keys.append(key, data_dict[key][0], data_dict[key][1])

    for item in final_keys:
        keyword_df.append(item)

    keyword_df.to_csv('keyword.csv')
    print("Testing done!!!")
    return


if __name__ == "__main__":
    path_name = "reviews_Electronics_5.json.gz"
    testing_function(path_name)



