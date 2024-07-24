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
