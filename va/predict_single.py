import pandas as pd
from data_manager import DataManager
from agent.vision_assistant_agent import VisionAssistant
import time

if __name__ == '__main__':
    # public set
    csv_path = f'../data/release_public_set.csv'
    data_root = '..'
    output_root = '../output'
    
    db = pd.read_csv(csv_path)
    db = DataManager(db, data_root)
    # db = DataManager(db, data_root)
    va = VisionAssistant(debug=True, timeout=50, output_root=output_root, is_thread = True, memory_limit_mb = 150)
    tic = time.time()
    for messages, artifact, ob in db:
        # if ob['id'] == "c1ada5be-4ab9-43c1-8742-8e1e384f7df1": # bear
        # if ob['id'] == "6ac4a782-b549-4c99-8c03-590d89afc289": # bird
        if ob['id'] == "8cda5be1-f41a-49b6-afe8-f87687de4bf9": # bicycle
            # result = va.predict(messages, artifact)
            va.add_task(messages, artifact, ob)
    va.start_task(1)
    toc=  time.time()
    print(f'Done in {round(toc-tic, 3)} sec.')
