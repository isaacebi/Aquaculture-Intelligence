import cv2
import numpy as np
import pandas as pd
import pafy
import os

class MotionGenerator:
    def __init__(self, save_path, duration, interval_frame: int = 15):
        self.save_path = save_path
        self.duration = duration
        self.interval_frame = interval_frame

    def random_motion_time(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if required columns exist
        required_columns = ['time_start', 'time_end']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Copy
        df = df.copy()

        # Convert time_start and time_end to seconds
        df['start_seconds'] = df['time_start'].apply(lambda x: pd.Timedelta(x).seconds)
        df['end_seconds'] = df['time_end'].apply(lambda x: pd.Timedelta(x).seconds)
        df['mid_seconds'] = df['start_seconds'] + ((df['end_seconds']-df['start_seconds'])/2)

        
        # Generate random seconds between start and end
        df['random_seconds'] = np.random.uniform(df['start_seconds'], df['end_seconds'] - self.duration)
        
        # Convert random seconds to time format
        df['random_time'] = pd.to_datetime(df["random_seconds"], unit='s').dt.strftime("%H:%M:%S")
        
        return df['random_time']
    
    def get_frames(
            self, 
            video_path=os.path.join(os.getcwd(), 'data', 'raw', 'sample_test.mp4'),
            interval=15, 
            time_start=0, 
            time_end=60):
        
        # Open the video file or camera stream
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return False
        
        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the frame number based on the time
        start_frame = int(round(time_start) * fps)
        end_frame = int(round(time_end) * fps)
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read set frame
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print(f"Error: Could not read frame at {time_start} seconds.")
            cap.release()
            return False

        frames = []

        for frame_number in range(start_frame + 1, end_frame):
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
        
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Collect frames every interval
            if frame_number % self.interval_frame == 0:
                frames.append(gray)

        return frames, end_frame
        

    def motion_history_image(
            self, 
            gray_images: list,
            end_frame: int,
            file_name='experiment_abn_frame'):
        
        mhi = np.zeros_like(gray_images[0], dtype=np.uint8)

        steps_brightness = 100 // len(gray_images)
        tau = steps_brightness
        
        for i in range(len(gray_images)-1):
            # Frame subtraction
            frame = cv2.subtract(gray_images[i], gray_images[i+1])

            # Blurring
            frame = cv2.GaussianBlur(frame, (5,5), 0)

            # Threshold differences
            _, frame = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)

            # Draw to blank canvas
            mhi = cv2.add(mhi, frame//tau)

            tau += steps_brightness

        # Save the frame as an image
        parent_folder = f"{self.save_path}"
        folder_name = os.path.join(parent_folder, f"{self.duration}_seconds")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        output_path = os.path.join(folder_name, f"{file_name}_{end_frame}.png")
        print(f"Snapshot at {end_frame} frame saved to {output_path}")
        cv2.imwrite(output_path, mhi)
        

if __name__ == "__main__":
    gen_mot = MotionGenerator(save_path="test", duration=30, interval_frame=15)
    frames, end_frame = gen_mot.get_frames()

    
    gen_mot.motion_history_image(
        gray_images=frames,
        end_frame=end_frame
    )