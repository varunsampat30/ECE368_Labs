import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

import math
from decimal import Decimal

def estimator(file_list, dictionary_email):
    """
    Returns a parameters p_d, and q_d from the training set for a given
    file list
    
    Input
    -----
    Output
    ------
    
    """
    freq_dic = util.get_word_freq(file_list)
    # must account for words that we encounter at least once
    ### Laplace smoothing -> one occurrence of each word has already occured
    ### this implies => pd = (# occ of word+1)/(Total # of words in the bag+D),
    ### where D is the total number of words
    
    totalWordsinFiles = 0
    for word in freq_dic:
         # safety check to see if word occurs more than
         # once
        totalWordsinFiles += freq_dic[word]
    estimator_list = util.Counter()
    for word in dictionary_email:
        estimator_list[word] = ((freq_dic[word] + 1))/((totalWordsinFiles + 
                 len(dictionary_email)))
    ### doesn't violate the sum of prob. because 
    ### before any occurrences, each word in the dictionary has a uniform
    ### prob. of 1/D, where D is the total number of words. Sigma(1/D) over
    ### all D will be 1.
    return estimator_list
    
    
def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    probabilities_by_category = []
    ## make dictionary
    
    spam_words = []
    for file in file_lists_by_category[0]:
        words = util.get_words_in_file(file)
        spam_words.extend(words)
        
    ham_words = []
    for file in file_lists_by_category[1]:
        words = util.get_words_in_file(file)
        ham_words.extend(words)    
    dictionary_emails = list(set(ham_words+spam_words))
    
    probabilities_by_category.append(estimator(file_lists_by_category[0], 
                                               dictionary_emails))
    probabilities_by_category.append(estimator(file_lists_by_category[1], 
                                               dictionary_emails))
    return probabilities_by_category

def probability(freq_dic, word_probabilities, prior):
    
    result = 1
    numerator = 0
    denominator = 1
    pd_prod = []
    
    for word in freq_dic:
        if word in word_probabilities:
            numerator += freq_dic[word]
            val = Decimal(word_probabilities[word]**freq_dic[word])
            pd_prod.append(val)
            denominator = denominator * math.factorial(freq_dic[word])
    result = Decimal(prior) * (Decimal(math.factorial(numerator))/denominator)*np.prod(pd_prod)
    return result
    
    """
    result = np.log(prior)
    for word in freq_dic:
        result += freq_dic[word] * np.log(word_probabilities[word])
    return result
    """
    
def classify_new_email(filename, probabilities_by_category, prior_by_category, zeta):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution
    zeta: coefficient to change bias and influence decisions

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    fileList = list()
    fileList.append(filename)
    wordFrequencies = util.get_word_freq(fileList)
    
    spamProbability = probability(
            wordFrequencies, 
            probabilities_by_category[0], 
            prior_by_category[0])
    
    hamProbability = probability(
            wordFrequencies, 
            probabilities_by_category[1], 
            prior_by_category[1])
    
   # What is zeta? -> check code summary
    classify_result = list()
    if Decimal(2-zeta)*spamProbability >= Decimal(zeta)*hamProbability:
        classify_result.append('spam')
    else:
        classify_result.append('ham')
        
    logResult = list()
    logResult.append(np.log(float(spamProbability)))
    logResult.append(np.log(float(hamProbability)))
    classify_result.append(logResult)
    
    return classify_result

def testModel(zeta, test_folder, probabilities_by_category, priors_by_category):
    """
    ----
    Outputs: 2 element list error
    
    """
    performance_measures = np.zeros([2,2])
        # Classify emails from testing set and measure the performance
            # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # ^ type 1 error
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # ^ type 2 error
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category, zeta)
    
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    return 100*(totals - correct)/totals,(performance_measures[0,1], performance_measures[1,0])

def plot_error_results(zetas, error_1, error_2):
    plt.figure()
    plt.grid()
    plt.plot(zetas, error_1, 'r', label='Type error 1')
    plt.plot(zetas, error_2, 'b', label='Type error 2')
    plt.legend()
    plt.xlabel('Zeta')
    plt.ylabel('Error %')

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    zeta = 1
    error, incorrect = testModel(zeta, 
                                 test_folder, 
                                 probabilities_by_category, 
                                 priors_by_category)
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
# Type 1 error is defined as the event that a spam email is misclassified as ham
# Type 2 error is defined as the event that a ham email is misclassified as spam
    print("Error 1: ", error[0])
    print("Error 2: ", error[1])
    
    error_1_results = list()
    error_2_results = list()
    zetas = np.arange(0, 2.1, 0.1)
    incorrect_1_list = []
    incorrect_2_list = []
    for zeta in zetas:
        error, incorrect = testModel(zeta, test_folder, probabilities_by_category, priors_by_category)
        error_1_results.append(error[0])
        error_2_results.append(error[1])
        incorrect_1_list.append(incorrect[0])
        incorrect_2_list.append(incorrect[1])
        
    plot_error_results(zetas, error_1_results, error_2_results)
    plt.figure()
    plt.xlabel("Type 1 Errors")
    plt.ylabel("Type 2 Errors")
    plt.title("Type 1 vs Type 2 error")
    plt.scatter(incorrect_1_list, incorrect_2_list)
    plt.savefig("nbc.pdf")
    
        
    
    