import os
import cv2
import uuid
import argparse
import numpy as np
import pandas as pd

try:
    from set_path import GetPath
    from motion_generator import MotionGenerator
    print('Running through Python Script')
except:
    from src import GetPath
    print('Running through Jupyter Notebooks')


RANDOM_STATE = 42
SAMPLE_SIZE = 20

# Generate full 60 seconds
def generate_full_mhi():
    # Hard coded path
    DATA_FOLDER = GetPath().shared_data()
    LOCAL_DATA_FOLDER = GetPath().local_data()

    RANDOM_TIME_FOLDER = os.path.join(LOCAL_DATA_FOLDER, 'data', 'random_time')
    ABN_B1_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b1.csv')
    ABN_B2_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b2.csv')
    ABN_B3_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b3.csv')
    OUTPUT_PATH = os.path.join(DATA_FOLDER, 'preprocess', 'mhi')

    # hardcoded path
    ABNORMAL_VIDEO_PATHS = GetPath().abnormal_path()

    # Videos path
    abn_vids = GetPath().abnormal_path()

    #
    gen_mot = MotionGenerator(save_path=OUTPUT_PATH)

    for i in range(3):
        exp_name = f"ABN_B{i+1}"
        dfSample = pd.read_csv(eval(f"ABN_B{i+1}_TIME"))
        dfSample['experiment'] = exp_name
        dfSample['random_time_start'] = gen_mot.full_motion_time(
            dfSample=dfSample
        )
        
        # Time format to seconds, needed as opencv only read seconds
        dfSample['time_start_seconds'] = dfSample['time_start'].apply(lambda x: pd.Timedelta(x).seconds)
        dfSample['time_end_seconds'] = dfSample['time_end'].apply(lambda x: pd.Timedelta(x).seconds)

        dfSample.apply(lambda x: gen_mot.motion_history_image(
            video_path=ABNORMAL_VIDEO_PATHS[i],
            time_start=x['time_start_seconds'],
            time_end=x['time_end_seconds'],
            file_name=f"{x['experiment']}_{x['ABN']}"
        ), axis=1)


# Generate specific time duration mhi
def mhi_duration(duration: int = 5, iterGen: int = 5, interval_frame: int = 15):
    # Hard coded path
    DATA_FOLDER = GetPath().shared_data()
    LOCAL_DATA_FOLDER = GetPath().local_data()

    RANDOM_TIME_FOLDER = os.path.join(LOCAL_DATA_FOLDER, 'data', 'random_time')
    ABN_B1_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b1.csv')
    ABN_B2_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b2.csv')
    ABN_B3_TIME = os.path.join(RANDOM_TIME_FOLDER, 'abnormal_b3.csv')
    OUTPUT_PATH = os.path.join(DATA_FOLDER, 'preprocess', 'mhi')

    # hardcoded path
    ABNORMAL_VIDEO_PATHS = GetPath().abnormal_path()
    NORMAL_VIDEO_PATHS = GetPath().normal_path()


    # Initiate Motion Generator
    gen_mot = MotionGenerator(save_path=OUTPUT_PATH, duration=duration, interval_frame=interval_frame)


    for gen in range(iterGen):
        for i in range(3):
            exp_name = f"ABN_B{i+1}"
            df = pd.read_csv(eval(f"ABN_B{i+1}_TIME"))

            # Remove level 0 from 2h video
            dfSample = df.drop(index=df[df['ABN']==0].index)

            # Resampling
            dfSample = dfSample.groupby('ABN').apply(
                lambda x: x.sample(SAMPLE_SIZE, replace=True)
            ).reset_index(drop=True)

            # Message
            print(f"{len(dfSample)} picture will be generated for MHI with duration of {duration}")

            dfSample['experiment'] = exp_name
            dfSample['random_time_start'] = gen_mot.random_motion_time(
                df=dfSample
            )
            
            # Time format to seconds, needed as opencv only read seconds
            dfSample['time_start_seconds'] = dfSample['time_start'].apply(lambda x: pd.Timedelta(x).seconds)
            dfSample['time_end_seconds'] = dfSample['time_end'].apply(lambda x: pd.Timedelta(x).seconds)
            dfSample['random_time_start_seconds'] = dfSample['random_time_start'].apply(lambda x: pd.Timedelta(x).seconds)


            # Generate motion history image based on random start time between defined start time to defined end time minus duration
            dfSample.apply(lambda x: gen_mot.motion_history_image(
                gray_images = gen_mot.get_frames(
                    video_path = ABNORMAL_VIDEO_PATHS[i],
                    time_start = x['random_time_start_seconds'],
                    time_end = x['random_time_start_seconds'] + duration
                )[0],
                end_frame = gen_mot.get_frames(
                    video_path = ABNORMAL_VIDEO_PATHS[i],
                    time_start = x['random_time_start_seconds'],
                    time_end = x['random_time_start_seconds'] + duration
                )[1],
                file_name = f"{x['experiment']}_{x['ABN']}"
            ), axis=1)

            # Specific level 0 from normal video
            end_time = 12*60*60 - duration
            start_times = np.random.choice(np.arange(end_time), size=(20))

            for sample in range(SAMPLE_SIZE):
                gray_images, end_frame = gen_mot.get_frames(
                    video_path=NORMAL_VIDEO_PATHS[i],
                    time_start=start_times[sample],
                    time_end=start_times[sample] + duration
                )

                gen_mot.motion_history_image(
                    gray_images=gray_images,
                    end_frame=end_frame,
                    file_name=f"ABN_0"
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Motion History Generation based on input duration, iteration generation, and interval frame"
    )
    parser.add_argument("--duration", type=int, default=5, help="Enter the MHI generation duration")
    parser.add_argument("--generation", type=int, default=5, help="Enter the MHI number generation")
    parser.add_argument("--interval", type=int, default=15, help="Enter the interval frame for MHI")
    args = parser.parse_args()

    # reassign
    duration = args.duration
    generation = args.generation
    interval = args.interval

    mhi_duration(duration=duration, iterGen=generation, interval_frame=interval)