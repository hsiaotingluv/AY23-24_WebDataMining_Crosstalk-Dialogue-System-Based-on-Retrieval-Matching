import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import accuracy_score


def load_data(file_path, chunk_size=10000):
    """
    Load data from a JSON file in chunks.

    Parameters:
        file_path (str): Path to the JSON file containing the dataset.
        chunk_size (int, optional): The number of data points to be read in each chunk. Defaults to 10000.

    Yields:
        list: A chunk of the dataset, each chunk being a list of dictionaries, where each dictionary represents a data point.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]


def preprocess_data_for_gridsearch(data_chunks):
    """
    Preprocess the data for use in grid search by concatenating text pairs into single strings.

    Parameters:
        data_chunks (list of lists): A list where each element is a chunk of data. Each chunk is a list of dictionaries,
                                     with each dictionary representing a single dialogue entry containing 'src', 'choices', 
                                     and the index of the correct response ('pos_idx').

    Returns:
        tuple: A tuple containing two elements:
            - A list of concatenated strings, each combining a source text with one of its associated choices.
            - A list of labels (integers), where each label indicates whether the associated choice is the correct answer.
    """
    texts = []   # A list to store concatenated pairs of src and choices
    labels = []  # A list to store the labels (0 or 1) indicating if the choice is the correct answer

    for chunk in data_chunks:
        for item in chunk:
            for choice in item['choices']:
                # Concatenate src text and choice into a single string
                concatenated_text = item['src'] + " " + choice
                texts.append(concatenated_text)

                # Determine if this choice is the correct answer
                is_correct_choice = (choice == item['choices'][item['pos_idx']])
                labels.append(int(is_correct_choice))
    
    return texts, labels


# Define a pipeline with TF-IDF Vectorizer and SGDClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('sgd', SGDClassifier(loss='log_loss'))
])

# Define a grid of hyperparameters for GridSearchCV
param = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'sgd__alpha': [0.0001, 0.001, 0.01],
    'sgd__max_iter': [500, 1000, 1500],
    'sgd__penalty': ['l2', 'l1', 'elasticnet']
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param, n_iter=10, cv=KFold(5), scoring='accuracy', n_jobs=-1)

# Load and Preprocess Training Data
train_data = next(load_data('../train_fuse.json', chunk_size=10000))
train_texts, train_labels = preprocess_data_for_gridsearch([train_data])

# Perform Randomized Search
random_search.fit(train_texts, train_labels)

# Evaluate the Best Model on Validation Set
valid_data = next(load_data('../valid_fuse.json', chunk_size=10000))
valid_texts, valid_labels = preprocess_data_for_gridsearch([valid_data])

best_model = random_search.best_estimator_
valid_predictions = best_model.predict(valid_texts)
validation_accuracy = accuracy_score(valid_labels, valid_predictions)
print(f'Validation Accuracy: {validation_accuracy}')

# Evaluate the Best Model on Test Set
test_data = next(load_data('../test_fuse.json', chunk_size=10000))
test_texts, test_labels = preprocess_data_for_gridsearch([test_data])

test_predictions = best_model.predict(test_texts)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy}')
