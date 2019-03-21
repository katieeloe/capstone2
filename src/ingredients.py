import pandas as pd
import numpy as np
import json
import nltk
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

def text_to_json(file_path):

    recipe_file = open(file_path, 'r')
    full_recipes_txt = recipe_file.read()
    full_recipes_txt = full_recipes_txt.replace('}{"vegetarian' , '}|||{"vegetarian')
    full_recipes_lst = full_recipes_txt.split("|||")
    full_recipes_json = []
    for recipe in full_recipes_lst:
        full_recipes_json.append(json.loads(recipe))

    return full_recipes_json

def encode_targets(train_labels, num_per_label_train, test_labels, num_per_label_test):

    y = []
    for label in train_labels:
        i = 0
        while i <= num_per_label_train - 1:
            y.append(label)
            i += 1
    for label in test_labels:
        i = 0
        while i <= num_per_label_test - 1:
            y.append(label)
            i += 1

    encoder = preprocessing.LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    len_of_train = len(train_labels) * num_per_label_train

    y_train = y[ : len_of_train]
    y_test = y[len_of_train :]
    y_encoded_train = y_encoded[ : len_of_train]
    y_encoded_test = y_encoded[len_of_train :]

    return y_train, y_test, y_encoded_train, y_encoded_test, encoder

def isolate_ingredients(full_recipes_json):

    ingredients_lst = []
    for recipe in full_recipes_json:
        ingredients = []
        for ingredient in recipe['extendedIngredients']:
            ingredients.append(ingredient['name'])

        ingredients_lst.append(ingredients)

    return ingredients_lst

def remove_space_from_ingredients(ingredients_lst):

    no_space_ingredients_lst = []
    for recipe in ingredients_lst:
        ingredients = []
        for idx, ingredient in enumerate(recipe):
            ingredients.append(ingredient.lower().replace(" ", ""))
        no_space_ingredients_lst.append(ingredients)

    return no_space_ingredients_lst

def convert_ingredients_to_str(ingredients_lst):
    ingredients_strs = []
    for recipe in ingredients_lst:
        ingredients_strs.append(" ".join(recipe).lower())

    return ingredients_strs

def lemmatize_words(ingredients_strs):
    lemmatized_ingredients = []

    for recipe in ingredients_strs:
        lemmatized_ingredients.append(nltk.word_tokenize(recipe))

    lemmatized_ingredients_strs = []
    for recipe in lemmatized_ingredients:
        lemmatized_ingredients_strs.append(" ".join(recipe).lower())

    return lemmatized_ingredients_strs

def tfidf_vectorizer(ingredients_strs_train, ingredients_strs_test):

    tfidf = TfidfVectorizer(analyzer = 'word', stop_words = 'english', max_features = 300, min_df = 5)
    X_tfidf_train = tfidf.fit_transform(ingredients_strs_train)
    X_tfidf_test = tfidf.transform(ingredients_strs_test)

    return X_tfidf_train, X_tfidf_test, tfidf

def fit_naive_bayes(X_tfidf_train, y_encoded_train):

    clf = MultinomialNB()
    clf.fit(X_tfidf_train, y_encoded_train)

    return clf

def predict_naive_bayes(fit_model, X_tfidf_train, X_tfidf_test):

    train_predictions = fit_model.predict(X_tfidf_train)
    test_predictions = fit_model.predict(X_tfidf_test)

    return train_predictions, test_predictions

def score_naive_bayes(train_predictions, y_encoded_train, test_predictions, y_encoded_test):
    train_accuracy = np.mean(train_predictions == y_encoded_train)
    test_accuracy = np.mean(test_predictions == y_encoded_test)

    return train_accuracy, test_accuracy

def identify_incorrect_classifications(incorrect_prediction_idxs, full_recipes_json, predictions, y):

    for idx in incorrect_prediction_idxs:
        print(f"{full_recipes_json[idx]['title']} : predicted, actual - {encoder.inverse_transform(predictions)[idx]}, {y[idx]}")


if __name__ == "__main__":

    full_recipes_json_train = text_to_json('/home/katie/01-OneDrive/01_galvanize_dsi/capstones/02-capstone_2/capstone2/data/recipes_raw.txt')
    full_recipes_json_test = text_to_json('/home/katie/01-OneDrive/01_galvanize_dsi/capstones/02-capstone_2/capstone2/data/recipes_raw _test_data.txt')


    train_cuisines = ['african', 'chinese', 'japanese', 'korean', 'vietnamese', 'thai', 'indian', 'british',
    'irish', 'french', 'italian', 'mexican', 'spanish', 'middle+eastern', 'jewish', 'american', 'cajun',
    'southern', 'greek', 'german', 'nordic', 'eastern+european', 'caribbean', 'latin+american']

    #there weren't enough african recipes in the dataset to pull new recipes. Remove african from test encode_targets
    test_cuisines = ['chinese', 'japanese', 'korean', 'vietnamese', 'thai', 'indian', 'british',
    'irish', 'french', 'italian', 'mexican', 'spanish', 'middle+eastern', 'jewish', 'american', 'cajun',
    'southern', 'greek', 'german', 'nordic', 'eastern+european', 'caribbean', 'latin+american']

    y_train, y_test, y_encoded_train, y_encoded_test, encoder = encode_targets(train_cuisines, 20, test_cuisines, 10)

    #ingredients left as is, so ingredients with a space in them (i.e. "smoked paprika") will become separate tokens
    ingredients_lst_train = isolate_ingredients(full_recipes_json_train)
    ingredients_lst_test = isolate_ingredients(full_recipes_json_test)

    #spaces in indiviual ingredients removed to maintain full ingredient title when tokenized
    # no_space_ingredients_lst_train = remove_space_from_ingredients(ingredients_lst_train)
    # no_space_ingredients_lst_test = remove_space_from_ingredients(ingredients_lst_test)

    ingredients_strs_train = convert_ingredients_to_str(ingredients_lst_train)
    ingredients_strs_test = convert_ingredients_to_str(ingredients_lst_test)
    lemmatized_train = lemmatize_words(ingredients_strs_train)
    lemmatized_test = lemmatize_words(ingredients_strs_test)


    # no_space_ingredients_strs_train = convert_ingredients_to_str(no_space_ingredients_lst_train)
    # no_space_ingredients_strs_test = convert_ingredients_to_str(no_space_ingredients_lst_test

    X_tfidf_train, X_tfidf_test, tfidf = tfidf_vectorizer(lemmatized_train, lemmatized_test)

    #no_space_X_tfidf_train, no_space_X_tfidf_test, no_space_tfidf = tfidf_vectorizer(no_space_ingredients_strs_train, no_space_ingredients_strs_test)

    #Instantiate classifier object
    clf = fit_naive_bayes(X_tfidf_train, y_encoded_train)
    #no_space_clf = fit_naive_bayes(no_space_X_tfidf_train, y_encoded_train)

    #Predict classes
    train_predictions, test_predictions = predict_naive_bayes(clf, X_tfidf_train, X_tfidf_test)
    #no_space_train_predictions, no_space_test_predictions = predict_naive_bayes(no_space_clf, no_space_X_tfidf_train, no_space_X_tfidf_test)

    #Score model predictions
    train_accuracy, test_accuracy = score_naive_bayes(train_predictions, y_encoded_train, test_predictions, y_encoded_test)
    #no_space_train_accuracy, no_space_test_accuracy = score_naive_bayes(no_space_train_predictions, y_encoded_train, no_space_test_predictions, y_encoded_test)

    print("Spaces not Removed from Ingredients")
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    # print("Spaces Removed Ingredients")
    # print(f"Train Accuracy: {no_space_train_accuracy}")
    # print(f"Test Accuracy: {no_space_test_accuracy}")

    residuals = y_encoded_test - test_predictions
    # no_space_residuals = y_encoded_test - no_space_test_predictions

    incorrect_prediction_idxs = np.argwhere(residuals != 0).flatten()
    #no_space_incorrect_prediction_idxs = np.argwhere(residuals != 0).flatten()

    #print("Spaces not Removed from Ingredients")
    #identify_incorrect_classifications(incorrect_prediction_idxs, full_recipes_json_test, test_predictions, y_test)
