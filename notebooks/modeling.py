import pandas as pd
df = pd.read_csv("../datasets/dataset(08052022)-(1.02).csv")

from pycaret.classification import *

exp_clf = setup(
    data=df,
    target="depression",
    session_id=123,
    fold_shuffle=True,
    use_gpu=True
)

compare_models()