class Model():
    def __init__(self, temperture, modelName):
        self.name: str = "Model"
        
        if temperture is None:
            print(f"[{self.name}] Notice: 'temperture' is None. Defaulting to 0.")
            self.temperture = 0
        else:
            self.temperture = temperture
            print(f"[{self.name}] Log: 'temperture' set to {self.temperture}.")

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
    
    def printtemperture(self):
        print(f'temperture： {self.temperture}')
        return

    def getRes(self, prompt: str) -> str:
        return ""
    
    def getListRes(self, promptList: list) -> list:
        return []
    
    def getTokenLens(self, text) -> int:
        return 0