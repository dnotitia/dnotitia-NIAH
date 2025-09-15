import os

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI


class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0.1)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer partially aligns with the reference but has minor omissions.
                Score 10: The answer is aligned with the reference. No need to consider the information order and no need to be exact the same as the reference.
                Only respond with a numberical score! """}

    def __init__(self,
                 model_name: str,
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 api_key: str = None,
                 base_url: str = None,
                 question_asked: str = None, ):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")
        if not model_name:
            raise ValueError("model_name must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked
        self.base_url = base_url
        self.api_key = api_key

        self.evaluator = ChatOpenAI(model=self.model_name,
                                    api_key=self.api_key,
                                    base_url=self.base_url,
                                    **self.model_kwargs)

    def evaluate_response(self, response: str) -> int:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                evaluator = load_evaluator(
                    "labeled_score_string",
                    criteria=self.CRITERIA,
                    llm=self.evaluator,
                )

                eval_result = evaluator.evaluate_strings(
                    # The models response
                    prediction=response,

                    # The actual answer
                    reference=self.true_answer,

                    # The question asked
                    input=self.question_asked,
                )

                return int(eval_result['score'])
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Error occurred during evaluation: {str(e)}, retrying {retry_count} time(s)...")
                    import time
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    print(f"Error occurred during evaluation: {str(e)}, maximum retry attempts reached")
                    return -1
