from Model.Model import Model
from Model.ModelType import ModelType, get_model_map

class ModelFactory():
    """
    Factory class responsible for instantiating the appropriate language model.
    By centralizing the creation logic here, the main application doesn't need 
    to know the specific implementation details of each model.
    """
    def __init__(self):
        # Constructor currently requires no initialization logic.
        pass

    def buildModel(self, type: ModelType, *args, **kwargs) -> Model:
        """
        Creates and returns an instance of the specified model type.
        
        Args:
            type (ModelType): The enum identifier for the requested model.
            *args: Variable length argument list to pass to the model's constructor.
            **kwargs: Arbitrary keyword arguments (like a ModelConfig object) 
            to pass to the model's constructor.
            
        Returns:
            Model: An instantiated concrete model object, or None if the type is invalid.
        """
        
        model_map = get_model_map()
        modelcls = model_map.get(type)

        if not modelcls:
            print("Error: Model doesn't exist!")
            return None
        
        return modelcls(*args, **kwargs)