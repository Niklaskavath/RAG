from collections.abc import Generator, Iterator
from functools import singledispatchmethod
from typing import Literal, overload

from ollama import ChatResponse, ResponseError, chat, show


class ModelNotFoundError(Exception):
    """
    Custom exception raised when the specified Ollama model is not found.
    """


class OllamaOfflineError(Exception):
    """
    Custom exception raised when Ollama is not running
    """


class ChatBot:
    """
    A class that represents a chat bot using the Ollama API.
    """

    # Attribues:
    model: str
    messages: list[dict[str, str]]

    def __init__(
        self,
        model_name: str = "llama3.2",
        system_instructions: str = "",
    ) -> None:
        """
        Initializes the chat bot.

        Args:
            model_name (string): The ollama model to use.

            system_instructions (string): Instructions for how the chat bot
                should behave.

        Raises:
            ModelNotFoundError: If provided model name is not reconized as an installed
                ollama model.

            OllamaOfflineError: If ollama is not installed or currently running.
        """

        self._ensure_model_install(model_name)

        self.model = model_name
        self.messages = [{"role": "system", "content": system_instructions}]

    @overload
    def ask(self, question: str, stream: Literal[True] = True) -> Generator[str]: ...

    @overload
    def ask(self, question: str, stream: Literal[False]) -> str: ...

    def ask(self, question: str, stream: bool = True) -> Generator[str] | str:
        """
        Prompts the model with a question. User can choose to receive the answer
        in a stream or as a complete string.

        Args:
            question (string): The question to prompt the model.

            stream (bool): Weather the answer should be streamed or given in
                full.

        Returns:
            Generator | string: The answer to the question, either as a
                stream or in full.
        """

        # Append the user question to the message history
        self.messages.append({"role": "user", "content": question})

        # Prompt model with the question and message history
        response: Iterator[ChatResponse] | ChatResponse
        try:
            if stream:
                response = chat(model=self.model, messages=self.messages, stream=True)
            else:
                response = chat(model=self.model, messages=self.messages, stream=False)
        except ConnectionError as e:
            raise OllamaOfflineError(
                "Cannot connect to Ollama. Ensure Ollama is installed and running"
            ) from e

        return self._generate_response(response)

    def _ensure_model_install(self, model: str) -> None:
        """
        Ensures that the provided model is installed and that ollama is running.

        Raises:
            ModelNotFoundError: If provided model name is not reconized as an installed
                ollama model.

            OllamaOfflineError: If ollama is not installed or currently running.
        """

        try:
            show(model)
        except ResponseError as e:
            raise ModelNotFoundError(
                f"'{model}' is not recognized as an Ollama model. "
                "Check if the specified model is installed."
            ) from e
        except ConnectionError as e:
            raise OllamaOfflineError(
                "Cannot connect to Ollama. Ensure Ollama is installed and running"
            ) from e

    @singledispatchmethod
    def _generate_response(self, response) -> Generator[str] | str:
        """
        Converts the response from ollama.chat() into either a streamed sequence of
        text chunks or a single complete string, depending on the input type.

        The response is appended to self.messages, either in full or as the iterator is
        consumed.

        Args:
            response (Iterator[ChatResponse] | ChatResponse): The output of
                ollama.chat().

        Returns:
            Generator[str] | string: Returns Generator if the input is a Iterator of
                ChatResponse, returns a string if the input is a ChatResponse object.

        Raises:
            TypeError: If input is neither ChatResponse nor Iterator of ChatResponse.
        """

        raise TypeError(
            f"Unsupported type: {type(response)}, expected "
            "Iterator[ChatResponse] or ChatResponse"
        )

    @_generate_response.register(Iterator)
    def _(self, response: Iterator[ChatResponse]) -> Generator[str]:
        # Append response placeholder to the message history
        self.messages.append({"role": "assistant", "content": ""})

        response_pos = len(self.messages) - 1

        # Define a generator function to yield text chunks from the response
        def generator_stream():
            for chunk in response:
                text_chunk = chunk["message"]["content"]

                self.messages[response_pos]["content"] += text_chunk

                yield text_chunk

        return generator_stream()

    @_generate_response.register
    def _(self, response: ChatResponse) -> str:
        response_text = response["message"]["content"]

        self.messages.append({"role": "assistant", "content": response_text})

        return response_text
