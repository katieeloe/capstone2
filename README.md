# Naive Bayes - Cuisine Classifier

Cuisines are not stagnant. While they have evolved over time (in addition to new cuisines being created), it is safe to say that historically, cuisines were built upon the ingredients and spices that were readily available in that region. Knowing that certain ingredients will be aligned with particular cuisines, can I build a classifier using NLP and relating models to correctly classify a recipe's cuisine based on the ingredient list.


## Data

The recipe data was gathered using a food API. The API had functionality allowing parameters filters (one of them being cuisine) to be passed with the request. This gave me the ability to ensure my classes were evenly weighted:

![](/home/katie/01-OneDrive/01_galvanize_dsi/capstones/02-capstone_2/capstone2/images/class_weights_train.png)

When I requested the test recipes, I offset my results by 10 pages to avoid duplicating recipes from my train set in my test set. Unfortunately, there weren't enough African recipes in the entire data set to pull any of this cuisine 10 pages deep. My resulting test set:

![](/home/katie/01-OneDrive/01_galvanize_dsi/capstones/02-capstone_2/capstone2/images/class_weights_test.png)


Each recipe was a group of nested dictionaries, formatted as a string. The recipes could be easily navigated by converting the strings to Json objects. The ingredients were contained in an inner dictionary called, 'extendedIngredients', sample below:

```python  
>>> full_recipes_json_train[16]['extendedIngredients']

[{'id': 1022020,
  'aisle': 'Spices and Seasonings',
  'image': 'garlic-powder.png',
  'consitency': 'solid',
  'name': 'garlic powder',
  'original': '1 teaspoon garlic powder',
  'originalString': '1 teaspoon garlic powder',
  'originalName': 'garlic powder',
  'amount': 1.0,
  'unit': 'teaspoon',
  'meta': [],
  'metaInformation': [],
  'measures': {'us': {'amount': 1.0,
    'unitShort': 'tsp',
    'unitLong': 'teaspoon'},
   'metric': {'amount': 1.0, 'unitShort': 'tsp', 'unitLong': 'teaspoon'}}}, ... (the rest of the ingredients)]
   ```

After digging through a handful of recipes, I chose to use the generic ['name'], which seemed to be the most general of the choices.
