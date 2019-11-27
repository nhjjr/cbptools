import pandas as pd


def get_participant_ids(file: str = 'participants.tsv', sep: str = '\t',
                        index_col: str = 'participant_id') -> list:
    participant_ids = pd.read_csv(
        file, sep=sep, engine='python'
    ).get(index_col).tolist()
    return participant_ids
