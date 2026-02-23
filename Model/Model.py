class Model():
    def __init__(self, tempature, modelName):
        self.name: str = "Model"
        
        if tempature is None:
            print(f"[{self.name}] Notice: 'tempature' is None. Defaulting to 0.")
            self.tempature = 0
        else:
            self.tempature = tempature
            print(f"[{self.name}] Log: 'tempature' set to {self.tempature}.")

        if modelName is None:
            print(f"[{self.name}] Notice: 'modelName' is None. Defaulting to ''.")
            self.modelName = ''
        else:
            self.modelName = modelName
            print(f"[{self.name}] Log: 'modelName' set to {self.modelName}.")

    def getName(self) -> str:
        return self.name
    
    def printName(self):
        print(f'Model： {self.name}')
        return
    
    def printTempature(self):
        print(f'Tempature： {self.tempature}')
        return

    def getRes(self, prompt: str) -> str:
        return ""
    
    def getListRes(self, promptList: list) -> list:
        return []
    
    def getTokenLens(self, text) -> int:
        return 0