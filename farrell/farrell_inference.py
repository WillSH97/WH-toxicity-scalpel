from Lexicon import CATEGORY_NAMES, CATEGORY_WORDS

def farrell_lexicon(input_str: str):
    '''
    function which returns all matched categories of Farrell Lexicon words based on input string.

    inputs:
    - input_str (str): I mean, guess lol

    outputs:
    - categories (list[int]): list of integers which map back to categories of misogyny 
    '''

    categories = []

    for category in list(CATEGORY_WORDS.keys()):
        if any(word in input_str.lower() for word in CATEGORY_WORDS[category]):
            categories.append(category)

    return categories