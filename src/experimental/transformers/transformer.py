import time, os, sys, importlib.util
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import numpy as np



def get_absolute_path():
    """
    This function return the absolute path of the file that is being executed
        1) It splits the path by the project name, which is äqevilatne to the parent level folder of the project.
        2) It appends the project name back (split() deletes it).
    """
    project_name = "ML-BD-for-code-quality"
    
    temp_path=os.path.realpath(__file__)
    
    splitted_path = temp_path.split(project_name)
    absolute_path = f"{splitted_path[0]}{project_name}"
    
    return absolute_path



def append_the_module_src():
    """This function adds the src folder as a module independenttly of the path/start point of the project"""
    absolute_path = get_absolute_path()
    sys.path.append(absolute_path)


# Adding the "src" module manuelly to prevent the error "ModuleNotFoundError: No module named 'src'". In case it couldn't be add automatically.
append_the_module_src()



from src.experimental.supervised.preprocessing import process_stack_trace_column, word2vec
from src.experimental.supervised.classifiers import rfc, svm, mlp, mlp_transf
from src.experimental.supervised.classifiers_hyper_parameter import rfc_hyper_parameter_grid, svm_hyper_parameter_grid, \
    mlp_hyper_parameter_grid, rfc_hyper_parameter_random, svm_hyper_parameter_random, mlp_hyper_parameter_random




def get_data_folder_path(data_folder):
    """ 
    This function returns the absolute path of the data file. It is independent of the starting point of the file.
        1) It gets the absolute path of the file that has been executed.
        2) It appends the path of the data folder ("data/supervised/")
    """
    absolute_path = get_absolute_path()

    final_path = f"{absolute_path}/data/{data_folder}/"
    return final_path



def read_data_file(data_folder: str, file_name: str, method: str):
    """
    This function return file as a pd.DataFrame and At the moment support methods are "pd.read_csv()" and "np.loadtxt()".
    Parameters:
        + file_name: the name of the file to read as a pd.DataFrame. Providing extension is optional and only supports csv at the moment (If there's no ".csv" it will append it to the file_name).
    """
    if not file_name:
        raise Exception(" ** [Error] Please provide a file name! ** ")
    
    if not data_folder:
        raise Exception(" ** [Error] Please provide the folder name of the data! ** ")
    
    data_path = get_data_folder_path(data_folder)
    
    final_path = f"{data_path}{file_name}"
    # Check if extention exists, if not append it
    if not file_name.endswith(".csv"):
        final_path = f"{data_path}{file_name}.csv"
        
    if method == "read_csv":
        df = pd.read_csv(final_path)
    elif method == "loadtxt":
        df = np.loadtxt(final_path, delimiter=',', dtype=float)
    else:
        raise Exception("[Error] Method not supported! Supported methodes are 'read_csv' and 'loadtxt'")
    return df



def transformer(data: list, model_name):
    model = SentenceTransformer(model_name, device='cpu')
    # Encode all sentences and return as numpy array
    data_embeddings = list()
    for item in data:
        item_embedding = model.encode(item, convert_to_numpy=True)
        data_embeddings.append(item_embedding)
    
    return data_embeddings



def transform_with_pretrained_models(df, col, models_list: list, export: bool = False):
    if export:
        if models_list == []:
            raise Exception("[Error] Please provide the pretrained models' names!")
        data = list(df[col])
        for model_name in models_list:
            print(f" **** Trasnforming {col} using the model: {model_name} **** ")
            
            embedding = transformer(data, model_name)
            df = pd.DataFrame(embedding)
                
            if col == "Stack trace":
                exported_filename = f"transformer_{model_name}_github_monkey.csv"
            elif col == "Code":
                exported_filename = f"transformer_{model_name}_language_dataset.csv"
            
            data_folder_path = get_data_folder_path(data_folder="transformer")
            path_for_exported_file = f"{data_folder_path}{exported_filename}"
            df.to_csv(path_for_exported_file, index=False, header=False)



def get_transformer_files():
    """ This function will collect the file names in a data/transformer """

    data_transformer_folder = get_data_folder_path("transformer")
    all_transformer_files = os.listdir(data_transformer_folder)
    return all_transformer_files



start = time.time()
print('started')

df_monkey_labeled = read_data_file(data_folder="supervised", file_name="monkey_selection", method="read_csv")
df_github_labeled = read_data_file("supervised", "github_selection", method="read_csv")

df_github_monkey = read_data_file("supervised", "monkey_processed", method="read_csv")
df_language_dataset = read_data_file("language_data", "10_language_dataset_with_groundtruth", method="read_csv")

models_list = ["paraphrase-multilingual-mpnet-base-v2", "paraphrase-MiniLM-L3-v2", "paraphrase-multilingual-MiniLM-L12-v2"]

data_dict = {
    "Stack trace": df_github_monkey, "Code": df_language_dataset
}

for col, df in data_dict.items():
    # "export=False" to not train the function if data is already transformed
    transform_with_pretrained_models(df=df, col=col, models_list=models_list, export=False)


# Preprocessing
print('preprocessing completed:', time.time() - start)

"""
#?: This should be not nessecary any more, because embedding is done by SentenceTransformer()
# Word-Embedding
v = TfidfVectorizer(use_idf=True)
tf_idf = v.fit_transform(df_github_monkey['Stack trace'])
# w2v = word2vec(df_github_monkey['Stack trace'])
"""

all_transformer_files = get_transformer_files()

for transformer_file in all_transformer_files:
    transformer_result = read_data_file(data_folder="transformer", file_name=transformer_file, method="loadtxt")
    # print(transformer_result)

    print('word-embedding completed:', time.time() - start)

    test_size = 0.3  # 70:30 split
    features = transformer_result
    # features = w2v
    
    # labels_dict = {"monkey": df_github_monkey['Exception name'], "Code": df_language_dataset['Language']}
    if "monkey" in transformer_file:
        labels = df_github_monkey['Exception name']
    else:
        labels = df_language_dataset['Language']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    print('dataset splitting completed:', time.time() - start)

    # Random Forest Classifier
    # rfc(start, X_train, X_test, y_train, y_test)

    # Support Vector Machine
    svm(start, X_train, X_test, y_train, y_test)

    # Neural Network for Transformer
    # mlp_transf(start, X_train, X_test, y_train, y_test, 500, 1000, 1)

    print('completed:', time.time() - start)
