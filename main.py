# main.py
from mouse import iniciar_listener, eventos
import pandas as pd

if __name__ == "__main__":
    iniciar_listener()

    df = pd.DataFrame(eventos)
    print(df)