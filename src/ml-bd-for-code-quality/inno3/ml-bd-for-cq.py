import PySimpleGUI as sg
import PySimpleGUI_Events as sge
import pandas as pd
from lib import process_stack_trace_column, word_embedding, k_means, rfc, svm, mlp
from sklearn.model_selection import train_test_split
import time

sg.theme("BlueMono")

# Keys
DATASET_KEY = "-DATASET-IN-"
LOAD_KEY = "-BTN-LOAD-"
FILE_NAME_KEY = "-FILE-OUT-"
COLUMN_NAME_KEY = "-COLS-OUT"
EXAMPLE_FRAME = "-FRAME-EXAMPLE-"
SHAPE_KEY = "-SHAPE-"
SETTINGS_FRAME = "-FRAME-SETTINGS-"
CLUSTERING_FRAME = "-FRAME-CLUSTERING-"
EXAMPLE_DATA_KEY = "-DATA-OUT-"
COL_DATA_KEY = "-COL-DATA-IN-"
COL_LABEL_TEXT = "-COL-LABEL-"
COL_LABEL_KEY = "-COL-LABEL-IN-"
IS_PREPROCESSED_KEY = "-CB-PREPROCESSED-"
SPECIAL_CHARS_KEY = "-CB-SPECIAL-CHARS-"
RADIO_LABELED_KEY = "-RAD-LABELED-"
RADIO_UNLABELED_KEY = "-RAD-UNLABELED-"
LOG_KEY = "-LOG-"
COMPUTE_KEY = "-BTN-COMPUTE-"
TRANSFORMER_KEY = "-USE-TRANSFORMER-"

# Keys for clustering algorithms
KMEANS_KEY = "-RAD-KMEANS-"
RFC_KEY = "-RAD-RFC-"
SVM_KEY = "-RAD-SVM-"
MLP_KEY = "-RAD-MLP-"

# Algorithm Parameters:
KMEANS_PARAMETER_FRAME = "-KMEANS-PARAMETER-FRAME-"
RFC_PARAMETER_FRAME = "-RFC-PARAMETER-FRAME-"
SVM_PARAMETER_FRAME = "-SVM-PARAMETER-FRAME-"
MLP_PARAMETER_FRAME = "-MLP-PARAMETER-FRAME-"

N_CLUSTERS = "-KMEANS-N-CLUSTERS-"
RANDOM_STATE = "-KMEANS-RANDOM-STATE-"
TEST_SIZE = "-TEST-SIZE-"
N_ESTIMATORS = "-RFC-ESTIMATORS-"
MAX_DEPTH = "-RFC-MAX-DEPTH-"
KERNEL_TYPE = "-SVM-KERNEL-TYPE-"
PCA_COMPONENTS = "-MLP-PCA-"
N_NEURONS = "-MLP-NEURONS-"
N_HIDDEN_LAYER = "-MLP-HIDDEN-LAYER-"

# Radio Groups
DATASET_TYPE = "dataset-type"
ALGORITHM = "algorithm"

# Algorithm specific parameter layouts:
kmeans_params = [
    [sg.Text("Number of Clusters: ", size=(15, 1)), sg.InputText(size=(5, 1), key=N_CLUSTERS, default_text=10)],
    [sg.Text("Random State: ", size=(15, 1)), sg.InputText(size=(5, 1), key=RANDOM_STATE, default_text=1)],
]

rfc_params = [
    [sg.Text("Test-Size:", size=(15, 1)), sg.Slider(range=(0, 1), orientation='horizontal', size=(15,4), default_value=0.3, resolution=0.01, key=TEST_SIZE)],
    [sg.Text("Estimators: ", size=(15, 1)), sg.InputText(size=(5, 1), key=N_ESTIMATORS, default_text=100)],
    [sg.Text("Max. Depth: ", size=(15, 1)), sg.InputText(size=(5, 1), key=MAX_DEPTH), sg.Text("(Leave blank for: max_depth=None)")],
]

svm_kernel_types = ['linear', 'poly', 'rbf']
svm_params = [
    [sg.Text("Test-Size:", size=(15, 1)), sg.Slider(range=(0, 1), orientation='horizontal', size=(15,4), default_value=0.3, resolution=0.01, key=TEST_SIZE)],
    [sg.Text("Kernel Type:", size=(15, 1)), sg.Combo(values=svm_kernel_types, readonly=True, size=(15,1), key=KERNEL_TYPE, default_value='linear')]
]

mlp_params = [
    [sg.Text("Test-Size:", size=(15, 1)), sg.Slider(range=(0, 1), orientation='horizontal', size=(15,4), default_value=0.3, resolution=0.01, key=TEST_SIZE)],
    [sg.Text("PCA Components: ", size=(15, 1)), sg.InputText(size=(5, 1), key=PCA_COMPONENTS, default_text=500)],
    [sg.Text("Neurons: ", size=(15, 1)), sg.InputText(size=(5, 1), key=N_NEURONS, default_text=1000)],
    [sg.Text("Hidden Layer: ", size=(15, 1)), sg.InputText(size=(5, 1), key=N_HIDDEN_LAYER, default_text=1)],
]

# Event Handlers
def _read_handler(values, application_data):
    file = values[DATASET_KEY]
    if file == "": return

    # display information about the dataset
    application_data.window[FILE_NAME_KEY].update(file)
    df = pd.read_csv(file, index_col=0)
    application_data.window[SHAPE_KEY].update(f"Rows x Columns: {df.shape[0]} x {df.shape[1]}")
    application_data.window[COLUMN_NAME_KEY].update(', '.join(df.columns.values))
    application_data.window[EXAMPLE_DATA_KEY].update(df.head(n=3))

    # show frames
    application_data.window[CLUSTERING_FRAME].update(visible=True)
    application_data.window[EXAMPLE_FRAME].update(visible=True)
    application_data.window[SETTINGS_FRAME].update(visible=True)

    # update dropdown menus with column
    application_data.window[COL_DATA_KEY].update(values=list(df.columns.values))
    application_data.window[COL_LABEL_KEY].update(values=list(df.columns.values))

    application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}File loaded successfully. [{file}]\n")

read_handler = sge.SimpleHandler(LOAD_KEY, _read_handler)

def _radio_labeled_handler(values, application_data):
    application_data.window[COL_LABEL_TEXT].update(visible=True)
    application_data.window[COL_LABEL_KEY].update(visible=True)
    application_data.window[KMEANS_KEY].update(disabled=True)
    application_data.window[RFC_KEY].update(disabled=False)
    application_data.window[SVM_KEY].update(disabled=False)
    application_data.window[MLP_KEY].update(disabled=False)

radio_labeled_handler = sge.SimpleHandler(RADIO_LABELED_KEY, _radio_labeled_handler)

def _radio_unlabeled_handler(values, application_data):
    application_data.window[COL_LABEL_TEXT].update(visible=False)
    application_data.window[COL_LABEL_KEY].update(visible=False)
    application_data.window[KMEANS_KEY].update(disabled=False)
    application_data.window[RFC_KEY].update(disabled=True)
    application_data.window[SVM_KEY].update(disabled=True)
    application_data.window[MLP_KEY].update(disabled=True)

radio_unlabeled_handler = sge.SimpleHandler(RADIO_UNLABELED_KEY, _radio_unlabeled_handler)

def _kmeans_parameter_handler(values, application_data):
    application_data.window[KMEANS_PARAMETER_FRAME].update(visible=True)
    application_data.window[RFC_PARAMETER_FRAME].update(visible=False)
    application_data.window[SVM_PARAMETER_FRAME].update(visible=False)
    application_data.window[MLP_PARAMETER_FRAME].update(visible=False)

kmeans_parameter_handler = sge.SimpleHandler(KMEANS_KEY, _kmeans_parameter_handler)

def _rfc_parameter_handler(values, application_data):
    application_data.window[KMEANS_PARAMETER_FRAME].update(visible=False)
    application_data.window[RFC_PARAMETER_FRAME].update(visible=True)
    application_data.window[SVM_PARAMETER_FRAME].update(visible=False)
    application_data.window[MLP_PARAMETER_FRAME].update(visible=False)

rfc_parameter_handler = sge.SimpleHandler(RFC_KEY, _rfc_parameter_handler)

def _svm_parameter_handler(values, application_data):
    application_data.window[KMEANS_PARAMETER_FRAME].update(visible=False)
    application_data.window[RFC_PARAMETER_FRAME].update(visible=False)
    application_data.window[SVM_PARAMETER_FRAME].update(visible=True)
    application_data.window[MLP_PARAMETER_FRAME].update(visible=False)

svm_parameter_handler = sge.SimpleHandler(SVM_KEY, _svm_parameter_handler)

def _rfc_parameter_handler(values, application_data):
    application_data.window[KMEANS_PARAMETER_FRAME].update(visible=False)
    application_data.window[RFC_PARAMETER_FRAME].update(visible=False)
    application_data.window[SVM_PARAMETER_FRAME].update(visible=False)
    application_data.window[MLP_PARAMETER_FRAME].update(visible=True)

mlp_parameter_handler = sge.SimpleHandler(MLP_KEY, _rfc_parameter_handler)

def _compute_handler(values, application_data):
    print(values, application_data)
    application_data.window[LOG_KEY].update("")
    start = time.time()
    # get values from input fields and settings
    file = values[DATASET_KEY]
    is_processed = values[IS_PREPROCESSED_KEY]
    replace_special_chars = values[SPECIAL_CHARS_KEY]
    use_transformer = values[TRANSFORMER_KEY]
    data_col = values[COL_DATA_KEY]
    label_col = values[COL_LABEL_KEY]

    # input validation
    if data_col == "" or (values[RADIO_LABELED_KEY] and label_col==""): return

    # read dataframe
    df = pd.read_csv(file, index_col=0)
    print(df)

    # preprocessing
    application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Preprocessing started. {time.time()-start}\n")
    processed_df = df if is_processed else process_stack_trace_column(df, data_col, replace_special_chars)
    application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Preprocessing finished. {time.time()-start}\n")
    print(processed_df)

    # word embedding
    application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Word-Embedding started. {time.time()-start}\n")
    tf_idf = word_embedding(processed_df, data_col)
    application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Word-Embedding finished. {time.time()-start}\n")
    print(tf_idf)

    # clustering
    if values[KMEANS_KEY]:
        # K-Means
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering started. [K-Means] {time.time()-start}\n")
        result = k_means(tf_idf, n_clusters=5)
        print(result)
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering finished. [K-Means] {time.time()-start}\n")

    elif values[RFC_KEY]:
        # Random Forest
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering started. [Random Forest Classifier] {time.time()-start}\n")
        result = rfc(tf_idf, df[label_col], test_size=0.3, estimators=100, max_depth=100)
        print(result)
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering finished. [Random Forest Classifier] {time.time()-start}\n")

    elif values[SVM_KEY]:
        # Support Vector Machine
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering started. [Support Vector Machine] {time.time()-start}\n")
        result = svm(tf_idf, df[label_col], test_size=0.3)
        print(result)
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering finished. [Support Vector Machine] {time.time()-start}\n")

    elif values[MLP_KEY]:
        # Neural Network
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering started. [Neural Network] {time.time()-start}\n")
        result = mlp(tf_idf, df[label_col], test_size=0.3, pca_components=500, neurons=1000, hidden_layer=1)
        print(result)
        application_data.window[LOG_KEY].update(f"{application_data.window[LOG_KEY].get()}Clustering finished. [Neural Network] {time.time()-start}\n")

compute_handler = sge.SimpleHandler(COMPUTE_KEY, _compute_handler)

# Layout
layout = [
    [
        sg.Input(key=DATASET_KEY),
        sg.FileBrowse("Browse", file_types=(("", "*.csv"),), initial_folder="../../../data", enable_events=True),
        sg.Button("Load", key=LOAD_KEY)
    ],
    [
        sg.Text("Selected File: "), sg.Text("No file selected", key=FILE_NAME_KEY)
    ],
    [
        sg.Frame("Example:", [
            [sg.Text("Rows x Columns:", key=SHAPE_KEY)],
            [sg.Text("Available Columns: "), sg.Text(key=COLUMN_NAME_KEY)],
            [sg.Text("Please read a .csv file to show the show the example structure of the data.", key=EXAMPLE_DATA_KEY)],
        ], key=EXAMPLE_FRAME, visible=False)
    ],
    [
        sg.Frame("Settings:", [
            [sg.Text("Already Preprocessed:  "), sg.Checkbox("", key=IS_PREPROCESSED_KEY)],
            [sg.Text("Replace Special Chars: "), sg.Checkbox("", key=SPECIAL_CHARS_KEY)],
            [
                sg.Text("Dataset Type: "),
                sg.Radio("Unlabeled", group_id=DATASET_TYPE, key=RADIO_UNLABELED_KEY, enable_events=True),
                sg.Radio("Labeled", group_id=DATASET_TYPE, key=RADIO_LABELED_KEY, enable_events=True)
            ],
            [sg.Text("Data Column:  "), sg.Combo(list(), key=COL_DATA_KEY, size=(20,1), readonly=True)],
            [sg.Text("Label Column: ", key=COL_LABEL_TEXT, visible=False), sg.InputCombo(list(), key=COL_LABEL_KEY, visible=False, size=(20,1), readonly=True)],
        ], key=SETTINGS_FRAME, visible=False)
    ],
    [
        sg.Frame("Clustering:", [
            [sg.Text("Use Transformer: "), sg.Checkbox("", key=TRANSFORMER_KEY)],
            [sg.Text("Select a clustering algorithm:")],
            [
                sg.Radio("K-Means", group_id=ALGORITHM, key=KMEANS_KEY, disabled=True),
                sg.Radio("Random Forest Classifier", group_id=ALGORITHM, key=RFC_KEY, disabled=True),
                sg.Radio("Support Vector Machine", group_id=ALGORITHM, key=SVM_KEY, disabled=True),
                sg.Radio("Neural Network", group_id=ALGORITHM, key=MLP_KEY, disabled=True),
             ],
            [sg.Frame("Parameters:", layout=kmeans_params, key=KMEANS_PARAMETER_FRAME, visible=True)],
            [sg.Frame("Parameters:", layout=rfc_params, key=RFC_PARAMETER_FRAME, visible=True)],
            [sg.Frame("Parameters:", layout=svm_params, key=SVM_PARAMETER_FRAME, visible=True)],
            [sg.Frame("Parameters:", layout=mlp_params, key=MLP_PARAMETER_FRAME, visible=True)],
            [sg.Button("Compute", key=COMPUTE_KEY)]
        ], key=CLUSTERING_FRAME, visible=False)
    ],
    [
        sg.Frame("Log:", [
            [sg.Text("Program started.\n", key=LOG_KEY)]
        ])
    ],
]

# Add events to Event Manager
event_manager = sge.EventManager()
event_manager += read_handler
event_manager += compute_handler
event_manager += radio_labeled_handler
event_manager += radio_unlabeled_handler
event_manager += kmeans_parameter_handler
event_manager += rfc_parameter_handler
event_manager += svm_parameter_handler
event_manager += mlp_parameter_handler

# Window
window = sg.Window("ML-BD for Code Quality", layout, resizable=True)
application_data = sge.SimpleApplicationState(window)

# Event Loop
while True:
    # read event
    event, values = window.read()

    # close window
    if event == sg.WINDOW_CLOSED: break

    # execute event
    event_manager.execute(event, values, application_data)

window.close()