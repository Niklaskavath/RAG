from collections.abc import Generator

import pytest
from ollama import ChatResponse, Message, ResponseError

from src.generation import ChatBot, ModelNotFoundError, OllamaOfflineError


@pytest.fixture
def chatbot(mocker) -> ChatBot:
    mocker.patch("src.generation.ChatBot._ensure_model_install")

    return ChatBot()


class TestChatBotRaises:
    def test_wrong_model_name_raises(self, mocker):
        mock = mocker.patch("src.generation.show")

        mock.side_effect = ResponseError("")

        with pytest.raises(ModelNotFoundError):
            ChatBot()

        mock.assert_called_once()

    def test_init_ollama_unavailable_raises(self, mocker):
        mock = mocker.patch("src.generation.show")

        mock.side_effect = ConnectionError("")

        with pytest.raises(OllamaOfflineError):
            ChatBot()

        mock.assert_called_once()

    def test_ask_ollama_unavailable_raises(self, chatbot, mocker):
        mock_chat = mocker.patch("src.generation.chat")

        mock_chat.side_effect = ConnectionError("")

        with pytest.raises(OllamaOfflineError):
            chatbot.ask("What is the capital of France?")

        mock_chat.assert_called_once()


class TestAsk:
    # ----------------------------------------------------------------------------------
    # Fixtures
    # ----------------------------------------------------------------------------------

    @pytest.fixture
    def mock_chat_unstreamed(self, mocker):
        mock_chat = mocker.patch("src.generation.chat")

        mock_chat.return_value = ChatResponse(
            message=Message(role="assistant", content="The capital of France is Paris.")
        )

        return mock_chat

    @pytest.fixture
    def mock_chat_streamed(self, mocker):
        mock_chat = mocker.patch("src.generation.chat")

        # Inner Generator to mock streamed response
        def streamed_response(ans: list[str]) -> Generator[ChatResponse]:

            for content in ans:
                yield ChatResponse(message=Message(role="assistant", content=content))

        mock_chat.side_effect = [
            streamed_response(["The capital of ", "France is Paris."]),
            streamed_response(["The capital of ", "Germany is Berlin."]),
        ]

        return mock_chat

    # ----------------------------------------------------------------------------------
    # Tests for stream = False
    # ----------------------------------------------------------------------------------

    def test_ask_returns_string(self, chatbot, mock_chat_unstreamed):
        response = chatbot.ask("What is the capital of France?", stream=False)

        mock_chat_unstreamed.assert_called_once()
        assert isinstance(response, str)

    def test_ask_unstreamed_updates_history(self, chatbot, mock_chat_unstreamed):
        question = "What is the capital of France?"

        _ = chatbot.ask(question, stream=False)

        mock_chat_unstreamed.assert_called_once()

        # Messages should contain the question, the answer, and the system message
        assert len(chatbot.messages) == 3

        assert chatbot.messages[-1]["role"] == "assistant"
        assert chatbot.messages[-1]["content"] == "The capital of France is Paris."

        assert chatbot.messages[-2]["role"] == "user"
        assert chatbot.messages[-2]["content"] == question

    # ----------------------------------------------------------------------------------
    # Tests for stream = True
    # ----------------------------------------------------------------------------------

    def test_ask_returns_generator(self, chatbot, mock_chat_streamed):
        response = chatbot.ask("What is the capital of France?", stream=True)

        mock_chat_streamed.assert_called_once()
        assert isinstance(response, Generator)

    def test_ask_streamed_yields_strings(self, chatbot, mock_chat_streamed):
        response = chatbot.ask("What is the capital of France?", stream=True)

        mock_chat_streamed.assert_called_once()

        for chunk in response:
            assert isinstance(chunk, str)

    def test_ask_streamed_updates_history(self, chatbot, mock_chat_streamed):
        first_question = "What is the capital of France?"
        second_question = "What is the capital of Germany?"

        first_response = chatbot.ask(first_question, stream=True)
        second_response = chatbot.ask(second_question, stream=True)

        assert mock_chat_streamed.call_count == 2

        # Consume the generators to ensure that the messages are updated
        for _ in first_response:
            pass
        for _ in second_response:
            pass

        # Messages should contain the two questions, their answers, and the system
        # message
        assert len(chatbot.messages) == 5

        # Assert that the messages are in the correct order and have the correct content
        assert chatbot.messages[-1]["role"] == "assistant"
        assert chatbot.messages[-1]["content"] == "The capital of Germany is Berlin."

        assert chatbot.messages[-2]["role"] == "user"
        assert chatbot.messages[-2]["content"] == second_question

        assert chatbot.messages[-3]["role"] == "assistant"
        assert chatbot.messages[-3]["content"] == "The capital of France is Paris."

        assert chatbot.messages[-4]["role"] == "user"
        assert chatbot.messages[-4]["content"] == first_question

    def test_ask_streamed_updates_history_while_streaming(
        self, chatbot, mock_chat_streamed
    ):
        response = chatbot.ask("What is the capital of France?", stream=True)

        mock_chat_streamed.assert_called_once()

        key = ["The capital of ", "France is Paris."]
        for i, chunk in enumerate(response):
            assert chunk == key[i]
            assert chatbot.messages[-1]["content"] == "".join(key[: i + 1])


class TestGenerateResponse:
    # ----------------------------------------------------------------------------------
    # Fixtures
    # ----------------------------------------------------------------------------------

    @pytest.fixture
    def response(self) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="The capital of France is Paris.")
        )

    @pytest.fixture
    def streamed_response(self) -> Generator[ChatResponse]:

        def generator() -> Generator[ChatResponse]:
            yield ChatResponse(
                message=Message(role="assistant", content="The capital of ")
            )

            yield ChatResponse(
                message=Message(role="assistant", content="France is Paris.")
            )

        return generator()

    # ----------------------------------------------------------------------------------
    # Tests for None input
    # ----------------------------------------------------------------------------------

    def test_generate_response_raises(self, chatbot):
        with pytest.raises(TypeError):
            chatbot._generate_response(None)

    # ----------------------------------------------------------------------------------
    # Tests for ChatResponse input
    # ----------------------------------------------------------------------------------

    def test_generate_response_returns_string(self, chatbot, response):
        gen_response = chatbot._generate_response(response)
        assert isinstance(gen_response, str)

    def test_generate_response_updates_history(self, chatbot, response):
        _ = chatbot._generate_response(response)

        assert chatbot.messages[-1]["role"] == "assistant"
        assert chatbot.messages[-1]["content"] == "The capital of France is Paris."

    # ----------------------------------------------------------------------------------
    # Tests for Generator input
    # ----------------------------------------------------------------------------------

    def test_generate_response_returns_generator(self, chatbot, streamed_response):
        gen_response = chatbot._generate_response(streamed_response)
        assert isinstance(gen_response, Generator)

    def test_generate_response_yields_strings(self, chatbot, streamed_response):
        gen_response = chatbot._generate_response(streamed_response)
        for chunk in gen_response:
            assert isinstance(chunk, str)

    def test_generate_response_stream_updates_history(self, chatbot, streamed_response):
        gen_response = chatbot._generate_response(streamed_response)

        # Consume the generator to ensure the messages are updates
        for _ in gen_response:
            pass

        assert chatbot.messages[-1]["role"] == "assistant"
        assert chatbot.messages[-1]["content"] == "The capital of France is Paris."

    def test_generate_response_updates_history_while_streaming(
        self, chatbot, streamed_response
    ):
        gen_response = chatbot._generate_response(streamed_response)

        key = ["The capital of ", "France is Paris."]
        for i, chunk in enumerate(gen_response):
            assert chunk == key[i]
            assert chatbot.messages[-1]["content"] == "".join(key[: i + 1])
