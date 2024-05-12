Support Vector Machines
Introduction
In this exercise, you will be using support vector machines (SVMs) to build a spam classifier. Before starting on the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook. The assignment can be promptly submitted to the coursera grader directly from this notebook (code and instructions are included below).

Before we begin with the exercises, we need to import all libraries required for this programming exercise. Throughout the course, we will be using numpy for all arrays and matrix operations, matplotlib for plotting, and scipy for scientific and numerical computation functions and tools..


1 Support Vector Machines
In the first half of this exercise, you will be using support vector machines (SVMs) with various example 2D datasets. Experimenting with these datasets will help you gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs. In the next half of the exercise, you will be using support vector machines to build a spam classifier.

1.1 Example Dataset 1
We will begin by with a 2D example dataset which can be separated by a linear boundary. The following cell plots the training data, which should look like this:


In this dataset, the positions of the positive examples (indicated with x) and the negative examples (indicated with o) suggest a natural separation indicated by the gap. However, notice that there is an outlier positive example x on the far left at about (0.1, 4.1). As part of this exercise, you will also see how this outlier affects the SVM decision boundary.



In this part of the exercise, you will try using different values of the  C parameter with SVMs. Informally, the  parameter is a positive value that controls the penalty for misclassified training examples. A large  C parameter tells the SVM to try to classify all the examples correctly. C  plays a role similar to , where  is the regularization parameter that we were using previously for logistic regression.

The following cell will run the SVM training (with C=1) using SVM software that we have included with the starter code (function svmTrain within the utils module of this exercise). When C=1, you should find that the SVM puts the decision boundary in the gap between the two datasets and misclassifies the data point on the far left, as shown in the figure (left) below.



In order to minimize the dependency of this assignment on external libraries, we have included this implementation of an SVM learning algorithm in utils.svmTrain. However, this particular implementation is not very efficient (it was originally chosen to maximize compatibility between Octave/MATLAB for the first version of this assignment set). If you are training an SVM on a real problem, especially if you need to scale to a larger dataset, we strongly recommend instead using a highly optimized SVM toolbox such as [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/). The python machine learning library [scikit-learn](http://scikit-learn.org/stable/index.html) provides wrappers for the LIBSVM library.


Your task is to try different values of  on this dataset. Specifically, you should change the value of  in the next cell to  and run the SVM training again. When , you should find that the SVM now classifies every single example correctly, but has a decision boundary that does not appear to be a natural fit for the data.





1.2 SVM with Gaussian Kernels
In this part of the exercise, you will be using SVMs to do non-linear classification. In particular, you will be using SVMs with Gaussian kernels on datasets that are not linearly separable.

1.2.1 Gaussian Kernel
To find non-linear decision boundaries with the SVM, we need to first implement a Gaussian kernel. You can think of the Gaussian kernel as a similarity function that measures the “distance” between a pair of examples, (
, 
). The Gaussian kernel is also parameterized by a bandwidth parameter, , which determines how fast the similarity metric decreases (to 0) as the examples are further apart. You should now complete the code in gaussianKernel to compute the Gaussian kernel between two examples, (
, 
). The Gaussian kernel function is defined as:




Once you have completed the function gaussianKernel the following cell will test your kernel function on two provided examples and you should expect to see a value of 0.324652.



From the figure, you can obserse that there is no linear decision boundary that separates the positive and negative examples for this dataset. However, by using the Gaussian kernel with the SVM, you will be able to learn a non-linear decision boundary that can perform reasonably well for the dataset. If you have correctly implemented the Gaussian kernel function, the following cell will proceed to train the SVM with the Gaussian kernel on this dataset.

You should get a decision boundary as shown in the figure below, as computed by the SVM with a Gaussian kernel. The decision boundary is able to separate most of the positive and negative examples correctly and follows the contours of the dataset well.




1.2.3 Example Dataset 3
In this part of the exercise, you will gain more practical skills on how to use a SVM with a Gaussian kernel. The next cell will load and display a third dataset, which should look like the figure below.


You will be using the SVM with the Gaussian kernel with this dataset. In the provided dataset, ex6data3.mat, you are given the variables X, y, Xval, yval.



Your task is to use the cross validation set Xval, yval to determine the best  and  parameter to use. You should write any additional code necessary to help you search over the parameters  and . For both  and , we suggest trying values in multiplicative steps (e.g., 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30). Note that you should try all possible pairs of values for  and  (e.g.,  and ). For example, if you try each of the 8 values listed above for  and for 
, you would end up training and evaluating (on the cross validation set) a total of 
 different models. After you have determined the best  and  parameters to use, you should modify the code in dataset3Params, filling in the best parameters you found. For our best parameters, the SVM returned a decision boundary shown in the figure below.





**Implementation Tip:** When implementing cross validation to select the best  and  parameter to use, you need to evaluate the error on the cross validation set. Recall that for classification, the error is defined as the fraction of the cross validation examples that were classified incorrectly. In `numpy`, you can compute this error using `np.mean(predictions != yval)`, where `predictions` is a vector containing all the predictions from the SVM, and `yval` are the true labels from the cross validation set. You can use the `utils.svmPredict` function to generate the predictions for the cross validation set.



The provided code in the next cell trains the SVM classifier using the training set  using parameters loaded from dataset3Params. Note that this might take a few minutes to execute.



2.1 Preprocessing Emails
Before starting on a machine learning task, it is usually insightful to take a look at examples from the dataset. The figure below shows a sample email that contains a URL, an email address (at the end), numbers, and dollar amounts.



While many emails would contain similar types of entities (e.g., numbers, other URLs, or other email addresses), the specific entities (e.g., the specific URL or specific dollar amount) will be different in almost every email. Therefore, one method often employed in processing emails is to “normalize” these values, so that all URLs are treated the same, all numbers are treated the same, etc. For example, we could replace each URL in the email with the unique string “httpaddr” to indicate that a URL was present.

This has the effect of letting the spam classifier make a classification decision based on whether any URL was present, rather than whether a specific URL was present. This typically improves the performance of a spam classifier, since spammers often randomize the URLs, and thus the odds of seeing any particular URL again in a new piece of spam is very small.

In the function processEmail below, we have implemented the following email preprocessing and normalization steps:

Lower-casing: The entire email is converted into lower case, so that captialization is ignored (e.g., IndIcaTE is treated the same as Indicate).

Stripping HTML: All HTML tags are removed from the emails. Many emails often come with HTML formatting; we remove all the HTML tags, so that only the content remains.

Normalizing URLs: All URLs are replaced with the text “httpaddr”.

Normalizing Email Addresses: All email addresses are replaced with the text “emailaddr”.

Normalizing Numbers: All numbers are replaced with the text “number”.

Normalizing Dollars: All dollar signs ($) are replaced with the text “dollar”.

Word Stemming: Words are reduced to their stemmed form. For example, “discount”, “discounts”, “discounted” and “discounting” are all replaced with “discount”. Sometimes, the Stemmer actually strips off additional characters from the end, so “include”, “includes”, “included”, and “including” are all replaced with “includ”.

Removal of non-words: Non-words and punctuation have been removed. All white spaces (tabs, newlines, spaces) have all been trimmed to a single space character.

The result of these preprocessing steps is shown in the figure below.

email cleaned

While preprocessing has left word fragments and non-words, this form turns out to be much easier to work with for performing feature extraction.

2.1.1 Vocabulary List
After preprocessing the emails, we have a list of words for each email. The next step is to choose which words we would like to use in our classifier and which we would want to leave out.

For this exercise, we have chosen only the most frequently occuring words as our set of words considered (the vocabulary list). Since words that occur rarely in the training set are only in a few emails, they might cause the model to overfit our training set. The complete vocabulary list is in the file vocab.txt (inside the Data directory for this exercise) and also shown in the figure below.

Vocab

Our vocabulary list was selected by choosing all words which occur at least a 100 times in the spam corpus, resulting in a list of 1899 words. In practice, a vocabulary list with about 10,000 to 50,000 words is often used. Given the vocabulary list, we can now map each word in the preprocessed emails into a list of word indices that contains the index of the word in the vocabulary dictionary. The figure below shows the mapping for the sample email. Specifically, in the sample email, the word “anyone” was first normalized to “anyon” and then mapped onto the index 86 in the vocabulary list.

word indices

Your task now is to complete the code in the function processEmail to perform this mapping. In the code, you are given a string word which is a single word from the processed email. You should look up the word in the vocabulary list vocabList. If the word exists in the list, you should add the index of the word into the word_indices variable. If the word does not exist, and is therefore not in the vocabulary, you can skip the word.

python tip: In python, you can find the index of the first occurence of an item in list using the index attribute. In the provided code for processEmail, vocabList is a python list containing the words in the vocabulary. To find the index of a word, we can use vocabList.index(word) which would return a number indicating the index of the word within the list. If the word does not exist in the list, a ValueError exception is raised. In python, we can use the try/except statement to catch exceptions which we do not want to stop the program from running. You can think of the try/except statement to be the same as an if/else statement, but it asks for forgiveness rather than permission.
An example would be:

try:
    do stuff here
except ValueError:
    pass
    # do nothing (forgive me) if a ValueError exception occured within the try statement
</div>

def processEmail(email_contents, verbose=True):




Once you have implemented processEmail, the following cell will run your code on the email sample and you should see an output of the processed email and the indices list mapping.



2.2 Extracting Features from Emails
You will now implement the feature extraction that converts each email into a vector in 
. For this exercise, you will be using n = # words in vocabulary list. Specifically, the feature 
 for an email corresponds to whether the 
 word in the dictionary occurs in the email. That is, 
 if the 
 word is in the email and 
 if the 
 word is not present in the email.

Thus, for a typical email, this feature would look like:

 
You should now complete the code in the function emailFeatures to generate a feature vector for an email, given the word_indices.

Once you have implemented emailFeatures, the next cell will run your code on the email sample. You should see that the feature vector had length 1899 and 45 non-zero entries.


2.3 Training SVM for Spam Classification
In the following section we will load a preprocessed training dataset that will be used to train a SVM classifier. The file spamTrain.mat (within the Data folder for this exercise) contains 4000 training examples of spam and non-spam email, while spamTest.mat contains 1000 test examples. Each original email was processed using the processEmail and emailFeatures functions and converted into a vector 
.

After loading the dataset, the next cell proceed to train a linear SVM to classify between spam () and non-spam () emails. Once the training completes, you should see that the classifier gets a training accuracy of about 99.8% and a test accuracy of about 98.5%.



Top Predictors for Spam
To better understand how the spam classifier works, we can inspect the parameters to see which words the classifier thinks are the most predictive of spam. The next cell finds the parameters with the largest positive values in the classifier and displays the corresponding words similar to the ones shown in the figure below.

our click remov guarante visit basenumb dollar pleas price will nbsp most lo ga hour
Thus, if an email contains words such as “guarantee”, “remove”, “dollar”, and “price” (the top predictors shown in the figure), it is likely to be classified as spam.

Since the model we are training is a linear SVM, we can inspect the weights learned by the model to understand better how it is determining whether an email is spam or not. The following code finds the words with the highest weights in the classifier. Informally, the classifier 'thinks' that these words are the most likely indicators of spam.



