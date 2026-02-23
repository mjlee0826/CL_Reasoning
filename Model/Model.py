class Model():
    def __init__(self, temperature, modelName):
        self.name: str = "Model"
        
        if temperature is None:
            print(f"[{self.name}] Notice: 'temperature' is None. Defaulting to 0.")
            self.temperature = 0
        else:
            self.temperature = temperature
            print(f"[{self.name}] Log: 'temperature' set to {self.temperature}.")

        if modelName is None:
            print(f"[{self.name}] Notice: 'modelName' is None. Defaulting to ''.")
            self.modelName = ''
        else:
            self.modelName = modelName
            print(f"[{self.name}] Log: 'modelName' set to {self.modelName}.")

    def getName(self) -> str:
        return self.name

    def getModelName(self) -> str:
        return self.modelName
    
    def printName(self):
        print(f'Model： {self.name}')
        return
    
    def printModelName(self):
        print(f'Model： {self.modelName}')
        return
    
    def printTemperature(self):
        print(f'temperature： {self.temperature}')
        return

    def getRes(self, prompt: str) -> str:
        return ""
    
    def getListRes(self, promptList: list) -> list:
        return []
    
    def getTokenLens(self, text) -> int:
        return 0