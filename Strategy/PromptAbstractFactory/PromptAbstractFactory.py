from Strategy.StrategyType import LanguageType

class PromptAbstractFactory:
    """
    Abstract base factory class for generating language-specific prompts.
    
    This class defines the interface for creating prompts in various languages.
    Concrete strategy prompt factories (like PromptTranslateFactory) should inherit 
    from this class and override the language-specific methods to provide the actual prompt templates.
    """
    def __init__(self):
        pass

    def englishPrompt(self):
        """
        Generates the prompt in English.
        Intended to be overridden by subclasses.
        """
        pass

    def chinesePrompt(self):
        """
        Generates the prompt in Chinese (Traditional/Simplified depending on subclass implementation).
        Intended to be overridden by subclasses.
        """
        pass

    def spanishPrompt(self):
        """
        Generates the prompt in Spanish.
        Intended to be overridden by subclasses.
        """
        pass

    def japanesePrompt(self):
        """
        Generates the prompt in Japanese.
        Intended to be overridden by subclasses.
        """
        pass

    def russianPrompt(self):
        """
        Generates the prompt in Russian.
        Intended to be overridden by subclasses.
        """
        pass

    def getPrompt(self, language_type: LanguageType, *args, **kwargs):
        """
        Routes the prompt generation request to the appropriate language method
        based on the provided LanguageType.
        
        Args:
            language_type (LanguageType): The target language for the prompt.
            *args: Variable length argument list containing prompt variables (e.g., the question).
            **kwargs: Arbitrary keyword arguments containing prompt variables.
            
        Returns:
            str or None: The generated prompt string, or None if the language is unsupported.
        """
        # Route to the correct method based on the Enum value.
        # Note: Aligned Enum members with LanguageType definition (e.g., CHINESE instead of ONLYCHINESE).
        if language_type == LanguageType.CHINESE:
            return self.chinesePrompt(*args, **kwargs)
        if language_type == LanguageType.ENGLISH:
            return self.englishPrompt(*args, **kwargs)
        if language_type == LanguageType.SPANISH:
            return self.spanishPrompt(*args, **kwargs)
        if language_type == LanguageType.JAPANESE:
            return self.japanesePrompt(*args, **kwargs)
        if language_type == LanguageType.RUSSIAN:
            return self.russianPrompt(*args, **kwargs)
        
        # Fallback for undefined languages
        return None