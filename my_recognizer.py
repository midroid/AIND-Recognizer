import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for test_word, (X, lengths) in test_set.get_all_Xlengths().items():
        probability = float("-inf")
        guess = ""
        probability_dictionary = {}

        for trained_word, model in models.items():
            try:
                calculated_probability = model.score(X, lengths)
                probability_dictionary[trained_word] = calculated_probability
            except:
                probability_dictionary[trained_word] = float("-inf")
            
            if calculated_probability > probability:
                probability = calculated_probability
                guess = trained_word
        probabilities.append(probability_dictionary)
        guesses.append(guess)
    # return probabilities, guesses

    return probabilities, guesses
    # raise NotImplementedError
