import os
import cv2
import uuid
import pandas as pd
import numpy as np

try:
    from set_path import GetPath
    from picture_generator import PictureGenerator
    print('Running through Python Script')
except:
    from src import GetPath
    print('Running through Jupyter Notebooks')


    
def main():
    # Hard coded path
    DATA_FOLDER = GetPath().shared_data()
    LOCAL_DATA_FOLDER = GetPath().local_data()

    RANDOM_TIME_FOLDER = os.path.join(LOCAL_DATA_FOLDER, 'data', 'random_time')
    ABN_B1_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b1.csv')
    ABN_B2_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b2.csv')
    ABN_B3_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b3.csv')
    OUTPUT_PATH = os.path.join(DATA_FOLDER, 'raw', 'abnormal')

    # Videos path
    abn_vids = GetPath().abnormal_path()

    #
    pic_gen = PictureGenerator(save_path=OUTPUT_PATH)

    for i in range(3):
        exp_name = f"ABN_B{i+1}"
        df = pd.read_csv(eval(f"ABN_B{i+1}_TIME"))
        df['experiment'] = exp_name
        df['random_time'] = pic_gen.time_to_snapshot(
            df=df
        )

        vid_path = abn_vids[i]
        df.apply(lambda x: pic_gen.take_snapshot(
            video_path = vid_path,
            file_name=f"{x['experiment']}_{x['ABN']}",
            time_seconds=int(x['random_seconds'])
        ), axis=1)

if __name__ == "__main__":
    main()