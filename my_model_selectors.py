import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score_bic = float("inf")
            best_model_bic = None

            for n in range(self.min_n_components, self.max_n_components + 1):
                bic_model = self.base_model(n)

                logL = bic_model.score(self.X, self.lengths)
                logN = np.log(len(self.X))

                # caluculating p(number of parameters) in Bayesian information criteria: BIC = -2 * logL + p * logN
                calculated_bic_score = n ** 2 + 2  (bic_model.n_features) * n - 1

                if calculated_bic_score < best_score_bic:
                    best_score_bic, best_model_bic = calculated_bic_score, bic_model
            return best_model_bic
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_DIC_score = float("-nf")
            best_DIC_model = None

            for n in range(self.min_n_components, self.max_n_components + 1):
                DIC_model = self.base_model(n)
                scores = []
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        scores.append(DIC_model.score(X, lengths))
                calculated_DIC_score = DIC_model.score(self.X, self.lengths) - np.mean(scores)
                if calculated_DIC_score > best_DIC_score:
                    best_DIC_score = calculated_DIC_score
                    best_DIC_model = DIC_model
            return best_DIC_model

        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_CV_score = float("inf")
            best_model_CV = None
            print('ok')

            for n in range(self.min_n_components, self.max_n_components+1):
                CV_scores = []
                split_method = KFold(n_splits=2)

                for train_idx, text_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(train_idx, self.sequences)

                    X, l = combine_sequences(test_idx, self.sequences)
                    CV_model = self.base_model(n)

                    CV_scores.append(CV_model.score(X, l)) 
                    
                    calculated_CV_score = np.mean(CV_scores)
                    
                    if calculated_CV_score < best_CV_score:
                        best_CV_score = calculated_CV_score
                        best_model_CV = CV_model
            
            return best_model_CV
        except:
            return self.base_model(self.n_constant)
