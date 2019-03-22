#spaces in indiviual ingredients removed to maintain full ingredient title when tokenized
# no_space_ingredients_lst_train = remove_space_from_ingredients(ingredients_lst_train)
# no_space_ingredients_lst_test = remove_space_from_ingredients(ingredients_lst_test)


# no_space_ingredients_strs_train = convert_ingredients_to_str(no_space_ingredients_lst_train)
# no_space_ingredients_strs_test = convert_ingredients_to_str(no_space_ingredients_lst_test


#no_space_X_tfidf_train, no_space_X_tfidf_test, no_space_tfidf = tfidf_vectorizer(no_space_ingredients_strs_train, no_space_ingredients_strs_test)

#no_space_clf = fit_naive_bayes(no_space_X_tfidf_train, y_encoded_train)

#no_space_train_predictions, no_space_test_predictions = predict_naive_bayes(no_space_clf, no_space_X_tfidf_train, no_space_X_tfidf_test)

#no_space_train_accuracy, no_space_test_accuracy = score_naive_bayes(no_space_train_predictions, y_encoded_train, no_space_test_predictions, y_encoded_test)

# print("Spaces Removed Ingredients")
# print(f"Train Accuracy: {no_space_train_accuracy}")
# print(f"Test Accuracy: {no_space_test_accuracy}")

# no_space_residuals = y_encoded_test - no_space_test_predictions

#no_space_incorrect_prediction_idxs = np.argwhere(residuals != 0).flatten()

#print("Spaces not Removed from Ingredients")
#identify_incorrect_classifications(incorrect_prediction_idxs, full_recipes_json_test, test_predictions, y_test)
