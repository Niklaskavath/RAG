import ollama
from collections.abc import Generator

class ChatBot:
    """
    A class that represents a chat bot using the Ollama API.
    """
    
    # Attribues:
    model: str
    messages: list[dict]
    
    def __init__(
            self, 
            model_name: str = "llama3.2",
            system_instructions: str = "",
        ) -> None:
        """
        Initializes the chat bot.
        
        Args:
            model_name (string): The ollama model to use.
            
            system_instructions (sting): Instructions for how the chat bot
                should behave.
        """
        
        # Check that ollama if ollama is live
        if not self._is_ollama_running():
            raise ConnectionError(
                "Failed to connect to Ollama. Check that Ollama is " +
                "Downloaded and running."
            )
        
        # Check that model is installed
        if not self._ensure_model_install(model_name):
            raise NameError(
                f"'{model_name}' is not recognized as an Ollama model. " +
                "Check if the specified model is installed."
            )
        
        self.model = model_name
        self.messages = [
            {
                "role" : "system",
                "content" : system_instructions
            }
        ]
    
    def ask(
            self, question: str, 
            stream: bool = True
        ) -> Generator[str, None, None] | str:
        """
        Promts the model with a question. User can choose to receive the answer 
        in a stream or as a complete sring.
        
        Args:
            question (sting): The question to prompt the model.
            
            stream (bool): Weather the answer should be streamed or given in
                full.
            
        Returns:
            generator | string: The answer to the question, either as a 
                stream or in full.
        """
        
        # Append the user question to the message history
        self.messages.append(
            {
                "role" : "user",
                "content" : question
            }
        )
        
        response = ollama.chat(
            model = self.model,
            messages = self.messages,
            stream = stream
        )
        
        # If answer is a stream, return the answer as a generator stream
        if stream: 
            return self._generate_stream(response) # type: ignore
        
        # If answer is given in full, return the full answer
        return self._generate_full(response) # type: ignore
    
    def _is_ollama_running(self) -> bool:
        """
        Private helper used to check if ollama is online.
        
        Returns:
            bool: True if ollama is online, otherwise False.
        """
        
        try:
            ollama.ps()
        except ConnectionError:
            return False
        return True
    
    def _ensure_model_install(self, model: str) -> bool:
        """
        Private helper used to check if ollama model is installed locally.
        
        Args:
            model (string): The ollama model to check for
            
        Returns:
            bool: True if model is installed on the system, otherwise False.
        """
        
        try:
            ollama.show(model)
        except ollama.ResponseError:
            return False
        return True
    
    def _generate_stream(
            self, 
            response: Generator[dict, None, None]
        ) -> Generator[str, None, None]:
        """
        Private helper that returns a textual response stream using the response
        generator from the ollama chat API. The response is appended to the 
        message history as it is generated.
        
        Args:
            response (generator): A generator stream of the response.
        
        Returns:
            generator: A generator stream of the textual response.
        """
        
        # Append an empty assistant response to the message history, which will
        # be filled with the response text as they are generated.
        self.messages.append(
            {
                "role" : "assistant",
                "content" : ""
            }
        )
        
        # Get the position of the assistant response in the message history
        # so that the text chunks can be appended to the correct message
        response_pos = len(self.messages) - 1
        
        # Return a generator that yields the text chunks as they are generated
        return self._response_chunk_generator(response, response_pos)
            
    def _response_chunk_generator(
            self,
            response: Generator[dict, None, None],
            response_pos: int
    ) -> Generator[str, None, None]:
        """
        Private helper that generates a textual response stream using response
        generator from the ollama chat API.
        
        Args:
            response (generator): A generator stream of the response.
            
            response_pos (int): The position of the assistant response in the
                message history so that the text chunks can be appended to the
                correct message.
                
        Returns:
            generator: A generator stream of the textual response.
        """    
        
        for chunk in response:
            text_chunk = chunk["message"]["content"]
            
            # Append the text chunk to the message corresponding to the question
            self.messages[response_pos]["content"] += text_chunk
            
            yield text_chunk

    def _generate_full(self, response: dict) -> str:
        """
        Private helper that adds the full textual response to the message 
        history and returns it.
        
        Args:
            response (dict): The response from the ollama chat API.
        
        Returns:
            string: The response to the question.
        """
        
        response_text = response["message"]["content"]
        
        # Append the user question and the assistant response to the message history
        self.messages.append(
            {
                "role" : "assistant",
                "content" : response_text
            }
        )
        
        return response_text