import cv2
import numpy as np
import pandas as pd
import pafy
import os

class MotionGenerator:
    def __init__(self, save_path):
        self.save_path = save_path

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
        df['random_seconds'] = np.random.uniform(df['start_seconds'], df['mid_seconds'])
        
        # Convert random seconds to time format
        df['random_time'] = pd.to_datetime(df["random_seconds"], unit='s').dt.strftime("%H:%M:%S")
        
        return df['random_time']

    def motion_history_image(
            self, 
            video_path=os.path.join(os.getcwd(), 'data', 'raw', 'sample_test.mp4'), 
            time_start=0, 
            time_end=60,
            file_name='experiment_abn_frame'):
                
        # Open the video file or camera stream
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return False
        
        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate Duration
        duration = time_end - time_start
        
        # Calculate the frame number based on the time
        start_frame = int(round(time_start) * fps)
        end_frame = int(round(time_end) * fps)
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read the first frame
        ret, prev_frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print(f"Error: Could not read frame at {time_start} seconds.")
            cap.release()
            return False
        
        # Convert the frame to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Create a motion history image
        h, w = prev_gray.shape[:2]
        mhi = np.zeros((h, w), dtype=np.float32)
        
        # Initialize the history frame
        history_frame = np.zeros_like(prev_gray, dtype=np.float32)

        timestamp = 0

        for frame_number in range(start_frame + 1, end_frame):
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % 15 == 0:
            
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate the absolute difference between frames
                frame_diff = cv2.absdiff(gray, prev_gray)
                
                # Threshold the difference
                _, motion_mask = cv2.threshold(frame_diff, 32, 1, cv2.THRESH_BINARY)

                timestamp += 1
                
                # Update history frame
                # history_frame = cv2.add(history_frame, motion_mask.astype(np.float32))
                cv2.motempl.updateMotionHistory(motion_mask, history_frame, timestamp, time_end-time_start)
                
                # Normalize history frame
                history_norm = cv2.normalize(history_frame, None, 0, 255, cv2.NORM_MINMAX)
                
                # Convert to uint8 for display
                history_display = history_norm.astype(np.uint8)
                
                # Display the result (optional)
                # cv2.imshow('Motion History Image', history_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Update previous frame
                prev_gray = gray
        
        # Save the frame as an image
        folder_name = f"{self.save_path}_{duration}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        output_path = os.path.join(folder_name, f"{file_name}_{frame_number}.png")
        cv2.imwrite(output_path, history_display)
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gen_mot = MotionGenerator(save_path="test")
    gen_mot.motion_history_image()