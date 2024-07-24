import os

class GetPath:
    def __init__(self):
        self.current_path = self.set_current_path()

    def data(self, fileName='data'):
        if fileName in os.listdir(self.current_path):
            target_folder = os.path.join(self.current_path, fileName)
            print(f"--- Defined Datapath \n{target_folder}")
            return target_folder
        
        print(f"--- Moving Project Path Upward")
        self.current_path = os.path.dirname(self.current_path)
        return self.data()
    
    def abnormal_path(self) -> list:
        # TODO: File structure need to be change in the future
        # Hard coded path
        ABNORMAL_FOLDER = "D:/david's experiment/manuel observation video"
        ABN_B1 = os.path.join(ABNORMAL_FOLDER, 'abnormal', 'B1', '2023-10-23 14-18-13.mp4')
        ABN_B2 = os.path.join(ABNORMAL_FOLDER, 'abnormal', 'B2', '2023-10-25 13-02-22.mp4')
        ABN_B3 = os.path.join(ABNORMAL_FOLDER, 'abnormal', 'B3', '2023-10-26 12-48-44.mp4')
        return [ABN_B1, ABN_B2, ABN_B3]
    
    def set_current_path(self):
        # Hard coded path
        LAB_PATH = "D:/fish_behavior"
        # Check if this run on lab
        if os.path.exists(LAB_PATH):
            return LAB_PATH
        
        return os.getcwd()


if __name__ == "__main__":
    pathing = GetPath()
    datapath = pathing.data()
