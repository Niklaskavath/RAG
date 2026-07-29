from ollama import (
    chat,
    ps,
    show,
    ChatResponse,
    ResponseError
)
from collections.abc import Generator, Iterator
from typing import Literal, overload
from functools import singledispatchmethod

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
    
    @overload
    def ask(
            self,
            question: str,
            stream: Literal[True] = True
    ) -> Generator[str, None, None]: ...
    
    @overload
    def ask(
            self,
            question: str,
            stream: Literal[False]
    ) -> str: ...
    
    def ask(
            self,
            question: str,
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
        
        # Prompt model with the question and message history
        response = chat(
            model = self.model,
            messages = self.messages,
            stream = stream
        )
        
        return self._generate_response(response)
    
    def _is_ollama_running(self) -> bool:
        """
        Private helper used to check if ollama is online.
        
        Returns:
            bool: True if ollama is online, otherwise False.
        """
        
        try:
            ps()
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
            show(model)
        except ResponseError:
            return False
        return True
    
    @singledispatchmethod
    def _generate_response(self, response) -> Generator[str, None, None] | str:
        raise TypeError(
            f"Unsupported type: {type(response)}, expected "
            "Iterator[ChatResponse] or ChatResponse"
        )
        
    @_generate_response.register(Iterator)
    def _(
        self,
        response: Iterator[ChatResponse]
    ) -> Generator[str, None, None]:
        # Append response placeholder to the message history
        self.messages.append(
            {
                "role" : "assistant",
                "content" : ""
            }
        )
        
        response_pos = len(self.messages) - 1
        
        # Define a generator function to yield text chunks from the response
        def generator_stream():
            for chunk in response:
                text_chunk = chunk["message"]["content"]
                
                # Append the text chunk to the response placeholder
                self.messages[response_pos]["content"] += text_chunk
                
                yield text_chunk
        
        return generator_stream()
    
    @_generate_response.register
    def _(
        self,
        response: ChatResponse
    ) -> str:
        response_text = response["message"]["content"]
        
        self.messages.append(
            {
                "role" : "assistant",
                "content" : response_text
            }
        )

        return response_text