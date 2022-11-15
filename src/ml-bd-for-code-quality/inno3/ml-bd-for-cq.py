import PySimpleGUI as gui

gui.theme("BlueMono")

upload = gui.Column([
    [gui.Frame("Dataset:", layout=[
        [gui.In(key="-DATA-IN-", size=(30, 1)), gui.FileBrowse(file_types=(("", "*.csv"),), initial_folder="../../../data", auto_size_button=True)],
        [
            gui.Text("Dataset labeled: "),
            gui.Radio("Yes", group_id="radio-labeled", key="-DATA-LABELED-YES-", size=(5, 1), enable_events=True),
            gui.Radio("No", group_id="radio-labeled", default=True, key="-DATA-LABELED-NO-", size=(5, 1), enable_events=True)
        ],
        [
            gui.Text("Dataset preprocessed: "),
            gui.Radio("Yes", group_id="radio-preprocessed", key="-DATA-PREPROCESSED-YES-", size=(5, 1), enable_events=True),
            gui.Radio("No", group_id="radio-preprocessed", default=True, key="-DATA-PREPROCESSED-NO-", size=(5, 1), enable_events=True)
        ],
        [
            gui.Text("Label of processed column: "),
            gui.InputText(key="-DATA-COLUMN-LABEL-IN-", size=(15, 1))
        ]
    ])]
])

algorithm = gui.Column([
    [gui.Frame("Clustering Algorithm: ", layout=[
        [
            gui.Radio("KMeans", group_id="radio-algorithm", key="-ALGORITHM-KMEANS-", size=(12,1), enable_events=True),
            gui.Radio("Neural Network", group_id="radio-algorithm", key="-ALGORITHM-NN-", size=(12, 1), enable_events=True),
            gui.Radio("Transformer", group_id="radio-algorithm", key="-ALGORITHM-TRANSFORMER-", size=(12, 1), enable_events=True),
        ]
    ])]
])

buttons = gui.Column([
    [gui.Frame("", layout=[
        [gui.Button("Cluster Dataset", key="-BTN-CLUSTER-", size=(50,2))],
    ])]
])

layout = [
    [gui.Text("Machine Learning & Big Data for Code Quality", size=(500, 1), justification="center", key="-TITLE-")],
    [upload],
    [algorithm],
    [buttons]
]

window = gui.Window("ML-BD for Code Quality", layout, size=(500, 400), resizable=True)

while True:
    # read event
    event, values = window.read()

    # close window
    if event == gui.WINDOW_CLOSED: break

window.close()