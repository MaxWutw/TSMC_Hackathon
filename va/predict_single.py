import pandas as pd
from data_manager import DataManager
from agent.vision_assistant_agent import VisionAssistant
import time

if __name__ == '__main__':
    csv_path = f'../data/release_public_set.csv'
    data_root = '..'
    output_root = '../output'

    db = pd.read_csv(csv_path)
    db = DataManager(db, data_root)
    # db = DataManager(db, data_root)
    va = VisionAssistant(debug=True, timeout=50, output_root=output_root, is_thread = True, memory_limit_mb = 150)
    tic = time.time()
    for messages, artifact, ob in db:
        if ob['id'] =="6370df4c-de52-4485-a798-b9a0d6eeb4a0":
            # result = va.predict(messages, artifact)
            va.add_task(messages, artifact, ob)
    va.start_task(1)
    toc=  time.time()
    print(f'Done in {round(toc-tic, 3)} sec.')