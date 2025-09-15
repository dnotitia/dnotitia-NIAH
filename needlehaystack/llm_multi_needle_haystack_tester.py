import asyncio
import glob
import json
import os
import time
from asyncio import Semaphore
from datetime import datetime, timezone

import numpy as np

from .evaluators import Evaluator
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .providers import ModelProvider


class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.
    
    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        model_to_test (ModelProvider): The model being tested.
        evaluator (Evaluator): The evaluator used to assess the model's performance.
        print_ongoing_status (bool): Flag to print ongoing status messages.
        eval_set (str): The evaluation set identifier.
        only_context (bool): Flag to indicate if only the context should be saved.
    """

    def __init__(self, *args,
                 needles=[],
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 print_ongoing_status=True,
                 eval_set="multi-needle-eval-sf",
                 case_name="",
                 only_context=False,
                 enable_dynamic_sleep=False,
                 base_sleep_time=2,
                 **kwargs):

        super().__init__(*args, model_to_test=model_to_test, **kwargs)
        self.needles = needles
        self.evaluator = evaluator
        self.evaluation_model = evaluator
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []
        self.case_name = case_name
        self.only_context = only_context
        self.enable_dynamic_sleep = enable_dynamic_sleep
        self.base_sleep_time = base_sleep_time

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at 
        designated depth percentages, effectively distributing these needles throughout the context. This method 
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text 
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens. 
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring 
        that the total token count (context plus needles) does not exceed the maximum allowable context length, 
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even 
        spacing for the remaining needles based on the remaining context length. It ensures that needles are 
        distributed as evenly as possible throughout the context after the first insertion. 
        
        Args:
            context (str): The original context string.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.
        
        Returns:
            str: The new context with needles inserted.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        tokens_context = self.ensure_complete_sentence(tokens_context)

        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
            tokens_context = self.ensure_complete_sentence(tokens_context)

        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)

        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = self.model_to_test.encode_text_to_tokens('.')

                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]

                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log 
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                print(
                    f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")

                # Adjust depth for next needle
                depth_percent += depth_percent_interval

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    def encode_and_trim(self, context, context_length):
        """
        Encodes the context to tokens and trims it to the specified length.
        
        Args:
            context (str): The context to encode and trim.
            context_length (int): The desired length of the context in tokens.
        
        Returns:
            str: The encoded and trimmed context.
        """
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

    async def generate_context(self, context_length, depth_percent):
        """
        Generates a context of a specified length and inserts needles at given depth percentages.
        
        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        
        Returns:
            str: The context with needles inserted.
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = await self.insert_needles(context, depth_percent, context_length)
        return context

    async def evaluate_and_log(self, context_length, depth_percent):
        """
        Evaluates the model's performance with the generated context and logs the results.
        
        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Generate context
        context = await self.generate_context(context_length, depth_percent)

        # Only execute sleep when TPM control is enabled
        if self.enable_dynamic_sleep:
            sleep_time = self.calculate_sleep_time(context_length)
            print(f"TPM control enabled: Waiting {sleep_time} seconds to avoid exceeding TPM limit...")
            await asyncio.sleep(sleep_time)

        test_start_time = time.time()
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Safely handle model_name
        model_name = self.model_name.replace(".", "_")
        model_name = model_name.replace("/", "_")

        # Build file name
        context_file_location = f'{model_name}_len_{context_length}_depth_{int(depth_percent)}'

        results = {
            'model': self.model_to_test.model_name,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needles': self.needles,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'file_name': context_file_location
        }

        # If context needs to be saved, execute regardless of whether only context is saved
        if self.save_contexts:
            # Handle file path modification
            base_path = './llm_multi_needle/contexts'
            model_folder = f'{self.case_name}/{model_name}'
            context_folder = os.path.join(base_path, model_folder)

            # Create directory
            if not os.path.exists(context_folder):
                os.makedirs(context_folder, exist_ok=True)

            # Build complete file path
            context_file_path = os.path.join(
                context_folder,
                f'{context_file_location}_context.txt'
            )

            try:
                print(f"Attempting to write to: {context_file_path}")
                with open(context_file_path, 'w', encoding='utf-8') as f:
                    f.write(context)
                print(f"Successfully wrote context to: {context_file_path}")
            except Exception as e:
                print(f"Error writing context file: {e}")
                print(f"Current working directory: {os.getcwd()}")

        # If not only saving context, execute evaluation part
        if not self.only_context:
            # LangSmith
            ## TODO: Support for other evaluators 
            if self.evaluator.__class__.__name__ == "LangSmithEvaluator":
                print("EVALUATOR: LANGSMITH")
                chain = self.model_to_test.get_langchain_runnable(context)
                self.evaluator.evaluate_chain(chain, context_length, depth_percent, self.model_to_test.model_name,
                                              self.eval_set, len(self.needles), self.needles,
                                              self.insertion_percentages)
                test_end_time = time.time()
                test_elapsed_time = test_end_time - test_start_time

            else:
                print("EVALUATOR: OpenAI Model")
                # Prepare your message to send to the model you're going to evaluate
                prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
                # Go see if the model can answer the question to pull out your random fact
                response = await self.model_to_test.evaluate_model(prompt)
                # Compare the reponse to the actual needle you placed
                score = self.evaluation_model.evaluate_response(response)

                test_end_time = time.time()
                test_elapsed_time = test_end_time - test_start_time

                results['model_response'] = response
                results['score'] = score

            self.testing_results.append(results)

            if self.print_ongoing_status:
                print(f"-- Test Summary -- ")
                print(f"Duration: {test_elapsed_time:.1f} seconds")
                print(f"Context: {context_length} tokens")
                print(f"Depth: {depth_percent}%")
                print(f"Score: {score}")
                print(f"Response: {response}\n")

            # Safely handle model_name
            model_name = self.model_name.replace(".", "_")
            model_name = model_name.replace("/", "_")
            print("model_name", model_name)

            # Ensure context_file_location also safely handles model name
            safe_model_name = self.model_name.replace(".", "_").replace("/", "_")
            context_file_location = f'{safe_model_name}_len_{context_length}_depth_{int(depth_percent * 100)}'

            if self.save_contexts:
                results['file_name'] = context_file_location

                # Handle file path modification
                base_path = './llm_multi_needle/contexts'
                model_folder = f'{self.haystack_dir}_{self.case_name}/{model_name}'
                context_folder = os.path.join(base_path, model_folder)

                # Use os.path.join to build path
                if not os.path.exists(context_folder):
                    os.makedirs(context_folder, exist_ok=True)

                # Build complete file path
                context_file_path = os.path.join(
                    context_folder,
                    f'{context_file_location}_context.txt'
                )

                try:
                    print(f"Attempting to write to: {context_file_path}")  # Debug info
                    with open(context_file_path, 'w', encoding='utf-8') as f:
                        f.write(context)
                    print(f"Successfully wrote context to: {context_file_path}")
                except Exception as e:
                    print(f"Error writing context file: {e}")
                    print(f"Current working directory: {os.getcwd()}")  # Debug info

            if self.save_results and not self.only_context:
                # Handle file path modification
                base_path = './llm_multi_needle/results'
                model_folder = f'{self.haystack_dir}_{self.case_name}/{model_name}'
                results_folder = os.path.join(base_path, model_folder)

                # Use os.path.join to build path
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder, exist_ok=True)

                # Build complete file path
                results_file_path = os.path.join(
                    results_folder,
                    f'{context_file_location}_results.json'
                )

                try:
                    print(f"Attempting to write results to: {results_file_path}")  # Debug info
                    with open(results_file_path, 'w') as f:
                        json.dump(results, f, ensure_ascii=False)
                    print(f"Successfully wrote results to: {results_file_path}")
                except Exception as e:
                    print(f"Error writing results file: {e}")
                    print(f"Current working directory: {os.getcwd()}")  # Debug info

            if self.seconds_to_sleep_between_completions:
                await asyncio.sleep(self.seconds_to_sleep_between_completions)

    async def bound_evaluate_and_log(self, sem, context_length, depth_percent):
        """
        Evaluation method with semaphore-controlled concurrency
        """
        try:
            async with sem:
                await self.evaluate_and_log(context_length, depth_percent)
        except Exception as e:
            print(f"Error occurred while processing context length {context_length}: {str(e)}")
            # Add extra wait time when error occurs
            await asyncio.sleep(10)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths  
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needles: {self.needles}")
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

    def ensure_complete_sentence(self, tokens_context):
        """Ensure context is truncated at complete sentence boundaries"""
        # Convert tokens back to text
        context = self.model_to_test.decode_tokens(tokens_context, preserve_sentence=True)

        # Find the position of the last complete sentence ending
        sentence_endings = ['. ', '! ', '? ']  # Define sentence ending markers
        last_end = -1
        for ending in sentence_endings:
            pos = context.rfind(ending)
            last_end = max(last_end, pos)

        if last_end == -1:
            return tokens_context  # If no sentence ending marker is found, return original tokens

        # Truncate text at the last complete sentence
        truncated_context = context[:last_end + 2]  # +2 to include period and space

        # Convert truncated text back to tokens
        return self.model_to_test.encode_text_to_tokens(truncated_context)

    def calculate_sleep_time(self, context_length):
        """
        Calculate wait time based on context length
        Args:
            context_length (int): Context length (number of tokens)
        Returns:
            float: Number of seconds to wait
        """
        # Base wait time (seconds)
        base_sleep = self.base_sleep_time

        # Set wait time multiplier based on context length tiers
        if context_length <= 4000:
            multiplier = 1
        elif context_length <= 8000:
            multiplier = 3
        elif context_length <= 16000:
            multiplier = 5
        elif context_length <= 32000:
            multiplier = 8
        elif context_length <= 64000:
            multiplier = 10
        elif context_length <= 128000:
            multiplier = 15
        else:
            multiplier = 20

        sleep_time = base_sleep * multiplier
        # Set maximum sleep time to 60 seconds
        return min(sleep_time, 60)
