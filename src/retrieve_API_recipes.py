import requests # Python package to send get/post requests to a webpage
import os       # Python package to interface with the operating system
from operator import itemgetter
from collections import Counter

def get_recipe_summaries(cuisine, number, offset):

    """
    The enpoint that allows a cuisine parameter to be specified returns a recipe summary, not the full recipe.
    Storing recipe summary to retreive recipe IDs, which will be fed into a second enpoint that returns full recipes.
    """

    url_start = 'https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/searchComplex?cuisine='
    url_cuisine = cuisine
    url_middle = '&type=main+course&ranking=2&limitLicense=false&offset='
    offset_num = str(offset)
    number = str(number)
    endpoint = url_start + url_cuisine + url_middle + offset_num + '&number=' + number
    payload = {'X-RapidAPI-Key': API_KEY}
    response = requests.get(endpoint, headers=payload)
    results = response.json()['results']
    for recipe_info in results:
        recipe_summaries.append(recipe_info)

def get_recipe_ids(results):
    recipe_ids = []
    for recipe in results:
        recipe_ids.append(str(recipe['id']))

    return recipe_ids

def get_full_recipes_batch(recipe_ids):
    """
    The batch endpoint was not working during capstone. Switched to looping through IDs and pulling indiviually.
    Keeping function for later use if endpoint issues are resolved
    """

    ids_str = '&2C'.join(recipe_ids)
    endpoint = 'https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/informationBulk?ids=' + ids_str
    payload = {'X-RapidAPI-Key': API_KEY}
    response = requests.get(endpoint, headers=payload)

    return response

def get_full_recipe_single(recipe_id):
    endpoint = 'https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/' + recipe_id + '/information'
    payload = {'X-RapidAPI-Key': API_KEY}
    response = requests.get(endpoint, headers=payload)

    return response

def write_to_file(response, file_name):
    file_path = '/home/katie/01-OneDrive/01_galvanize_dsi/capstones/02-capstone_2/capstone2/data/' + file_name
    with open(file_path, mode='a') as localfile:
        localfile.write(response)



if __name__ == "__main__":

    API_KEY = os.environ['Spoontacular_API_KEY']

    all_cuisines = ['african', 'chinese', 'japanese', 'korean', 'vietnamese', 'thai', 'indian', 'british',
    'irish', 'french', 'italian', 'mexican', 'spanish', 'middle+eastern', 'jewish', 'american', 'cajun',
    'southern', 'greek', 'german', 'nordic', 'eastern+european', 'caribbean', 'latin+american']

    #breaking recipe calls into two days, to stay within API limits
    monday_cuisines = ['african', 'chinese', 'japanese', 'korean', 'vietnamese', 'thai', 'indian', 'british',
    'irish', 'french', 'italian', 'mexican']

    tuesday_cuisines = ['spanish', 'middle+eastern', 'jewish', 'american', 'cajun',
    'southern', 'greek', 'german', 'nordic', 'eastern+european', 'caribbean', 'latin+american']

    #reduced number of cuisines to try to improve performance. Pulling aditional recipes from remaining cuisines to add to the train set.
    more_data_cuisines = ['chinese', 'korean', 'indian', 'british','irish', 'french', 'italian', 'mexican', 'middle+eastern', 'jewish', 'cajun', 'greek', 'german', 'caribbean']

    recipe_summaries = []
    for cuisine in more_data_cuisines:
        get_recipe_summaries(cuisine, number = 20, offset = 50)

    recipe_ids = get_recipe_ids(recipe_summaries)

    error_status_ids = []
    error_counts = Counter()

    text_files = ['extra_recipes.txt', 'recipes_raw.txt', 'recipes_raw _test_data.txt']

    for id in recipe_ids:
        response = get_full_recipe_single(id)
        if response.status_code != 200:
            error_status_ids.append((id , response.status_code))
            error_counts[response.status_code] += 1
            continue
        write_to_file(response.text, text_files[0])
