
def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.lower()
    wordlist = words.split()
    return wordlist
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    d={}
    from collections import Counter
# >>> z = ['blue', 'red', 'blue', 'yellow', 'blue', 'red']
# >>> Counter(z)
# Counter({'blue': 3, 'red': 2, 'yellow': 1})
    for sentence in messages:
        words = get_words(sentence)

        word_counts = Counter(words)
        for k,v in word_counts.items():
            if v>=5:
                d[k]= words.index(k)+v
    return d
            
        
        
        
        
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    npy = []
    for sentence in messages:
        words = get_words(sentence)
        from collections import Counter

        word_counts = dict(Counter(words))
        npy.append(np.array([count if word in word_dictionary else 0 for word,count in word_counts.items()]))
#          [x+1 if x >= 45 else x+5 for x in l]
    return np.array([npy])
        
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
#     print('matrix stuff ',matrix)
    phi_y1 = np.sum(labels) / len(labels)
    phi_y0 = len(labels) - phi_y1
    ones = np.sum(labels)
    zeros = 0
    phi_jy1 = 0
    phi_jy0 = 0
    
#     print('matrix',matrix)
    
#     for i in range(len(matrix)):
#         for j in range (len(labels[i])):
#             if labels[j]==1 and matrix[i]==1:
#                 phi_jy1 += 1
#                 ones += 1
#                 phi_jy1 = phi_jy1/ones

#             if labels[j]==0 and matrix[i]==1:
#                 phi_jy0 += 1
#                 zeros += 1
#                 phi_jy0 = phi_jy0/zeros

    phi_jy1 = np.sum(labels) / phi_y1
    phi_jy0 = np.sum(labels) / phi_y0
    theta = np.log(phi_y1 * phi_jy1 * phi_jy0 *phi_y0)
    model = matrix * theta
    return model
    
        
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
#     for i in range(len(matrix)):
#         if model[i]=
    print('matrix sum',(np.sum(matrix)).shape)
    print('matrix l',(len(matrix)).shape)
    phi_y1 = np.sum(matrix) / len(matrix)
    phi_y0 = len(matrix) - phi_y1
    phi_jy1 = np.sum(matrix) / phi_y1
    phi_jy0 = np.sum(matrix) / phi_y0
    print('phi_jy1', phi_jy1.shape)
    print('phi_y1', phi_y1.shape)
    print('phi_jy0', phi_jy0.shape)
    print('zeros', zeros.shape)
    y_hat = (phi_jy1*phi_y1)/((phi_jy1*phi_y1) + (phi_jy0*zeros))
    return y_hat
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    temp_numer = []
    temp_denom = []
    top5 = []
    for i in range (len(model)):
        if model[i] == 1:
            temp_numer.append(np.array([model[i]]))
        if model[i] == 0:
            temp_denom.append(np.array([model[i]]))
        top5[i] = np.append(np.log([temp_numer[i]/temp_denom[i]]))
    return np.array(top5)
            
        
        
    
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_radius = None
    for r in radius_to_consider:

        
        svm_predict = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, r)



        svm_accuracy = np.mean(svm_preds == val_labels)

        if best_radius is None:
            best_radius = (svm_acc, r)
  
    best_radius = max(best_radius, (svm_acc, r))

    return best_radius[1]
    # *** END CODE HERE ***


def main():
#     messages=["supp it's deji my name is deji cmon cuh its deji deji deji deji","khabib is the goat khabib smesh his face khabib is the best khabib khabib khabib","conor sucks conor lost conor tapped conor conor conor"]
#     cc=create_dictionary(messages)
#     tt = transform_text(messages,cc)
#     print(tt)
    
    train_messages, train_labels = util2.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util2.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util2.load_spam_dataset('spam_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    train_matrix = transform_text(train_messages, dictionary)
#     print('train matrix ',train_matrix[:100,:])
    fit_naive_bayes_model(train_matrix, train_labels)
    print('Size of dictionary: ', len(dictionary))

    util2.write_json('spam_dictionary', dictionary)

    
    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:],fmt = '%s')

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util2.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util2.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
