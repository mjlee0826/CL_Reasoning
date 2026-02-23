from Log.Log import Log

class OneAgentLog(Log):
    def __init__(self):
        super().__init__()
    
    def logInfo(self, strategy, model, dataset):
        print('=' * 30)
        print(f'Log Information')
        strategy.printName()
        model.printModelName()
        model.printTemperature()
        dataset.printName()
        dataset.printDataNums()
        print('=' * 30)