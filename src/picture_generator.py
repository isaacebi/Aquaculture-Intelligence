import os
import cv2
import uuid
import pandas as pd
import numpy as np

# Helper Function
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def str_to_hms(time_str):
    # h, m, s = map(int, time_str.str.split(':'))
    return time_str.str.split(":")


def seconds_to_time_string(seconds):
    hours = int(seconds // 3600)  # Get total hours
    seconds %= 3600  # Remaining seconds after calculating hours

    minutes = int(seconds // 60)  # Get total minutes
    seconds %= 60  # Remaining seconds after calculating minutes

    # Format time string with leading zeros for hours, minutes, and seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class PictureGenerator:
    def __init__(self, save_path) -> None:
        self.save_path = save_path
    
    def time_to_snapshot(
            self, 
            df: pd.DataFrame, 
            right_tail=False, 
            left_tail=True):
        
        # Check if required columns exist
        required_columns = ['time_start', 'time_end']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert time_start and time_end to seconds
        df['start_seconds'] = df['time_start'].apply(lambda x: pd.Timedelta(x).seconds)
        df['end_seconds'] = df['time_end'].apply(lambda x: pd.Timedelta(x).seconds)
        df['mid_seconds'] = df['start_seconds'] + ((df['end_seconds']-df['start_seconds'])/2)

        
        # Generate random seconds between start and end
        if right_tail:
            df['random_seconds'] = np.random.uniform(df['start_seconds'], df['mid_seconds'])
        if left_tail:
            df['random_seconds'] = np.random.uniform(df['mid_seconds'], df['end_seconds'])
        
        # Convert random seconds to time format
        df['random_time'] = pd.to_datetime(df["random_seconds"], unit='s').dt.strftime("%H:%M:%S")
        
        return df['random_time']


    def take_snapshot(self, video_path, file_name, time_seconds=0):
        """
        Captures a snapshot from a video file or camera stream at a specified time.
        
        Args:
        video_path (str or int): Path to video file or camera index.
        time_seconds (float): Time in seconds at which to capture the snapshot (default is 0).
        
        Returns:
        bool: True if snapshot was taken successfully, False otherwise.
        """
        # Open the video file or camera stream
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return False
        
        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the frame number based on the time
        frame_number = int(time_seconds * fps)
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print(f"Error: Could not read frame at {time_seconds} seconds.")
            cap.release()
            return False
        
        # Save the frame as an image
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        output_path = os.path.join(self.save_path, f"{file_name}_{frame_number}.png")
        cv2.imwrite(output_path, frame)
        
        # Release the video capture object
        cap.release()
        
        print(f"Snapshot at {time_seconds} seconds saved to {output_path}")
        return True