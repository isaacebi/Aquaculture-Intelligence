import os
import cv2
import uuid
import pandas as pd
import numpy as np

try:
    from set_path import GetPath
    from motion_generator import MotionGenerator
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
    OUTPUT_PATH = os.path.join(DATA_FOLDER, 'preprocess', 'mhi')

    # hardcoded path
    VIDEO_PATHS = GetPath().abnormal_path()

    # Videos path
    abn_vids = GetPath().abnormal_path()

    #
    gen_mot = MotionGenerator(save_path=OUTPUT_PATH)

    for i in range(3):
        exp_name = f"ABN_B{i+1}"
        df = pd.read_csv(eval(f"ABN_B{i+1}_TIME"))
        df['experiment'] = exp_name
        df['random_time_start'] = gen_mot.full_motion_time(
            df=df
        )
        
        # Time format to seconds, needed as opencv only read seconds
        df['time_start_seconds'] = df['time_start'].apply(lambda x: pd.Timedelta(x).seconds)
        df['time_end_seconds'] = df['time_end'].apply(lambda x: pd.Timedelta(x).seconds)

        df.apply(lambda x: gen_mot.motion_history_image(
            video_path=VIDEO_PATHS[i],
            time_start=x['time_start_seconds'],
            time_end=x['time_end_seconds'],
            file_name=f"{x['experiment']}_{x['ABN']}"
        ), axis=1)

if __name__ == "__main__":
    main()