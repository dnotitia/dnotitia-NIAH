import os
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import threading
from tqdm import tqdm
import argparse

# Set the working directory to the current file location
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import yaml
from needlehaystack.providers import OpenAI
from needlehaystack.evaluators import OpenAIEvaluator
from needlehaystack.llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from dotenv import load_dotenv

load_dotenv(".env")


# Load environment variables

def load_test_cases(file_path):
    """Load test case configuration file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_config(file_path: str) -> Dict[str, Any]:
    """Load model configuration file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_model_provider(model_name: str, model_config: Dict[str, Any]):
    """Get the corresponding Provider instance based on model name"""
    if model_name not in model_config['models']:
        raise ValueError(f"Model {model_name} configuration not found")

    config = model_config['models'][model_name]

    api_key = os.getenv(config['api_key'])
    base_url = os.getenv(config['base_url'])

    return OpenAI(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url
    )


def run_single_test(
        model_name: str,
        case_name: str,
        model_config: Dict,
        test_cases: Dict,
        context_lengths: List[int] = None,
        document_depth_percents: List[int] = None,
        num_concurrent_requests: int = 5,
        save_results: bool = True,
        save_contexts: bool = True,
        final_context_length_buffer: int = 300,
        print_ongoing_status: bool = True,
        only_context: bool = False,
        enable_dynamic_sleep: bool = True,
        base_sleep_time: float = 0.3,
        document_depth_percent_interval_type: str = 'linear',
        haystack_dir: str = "StarlightHarryPorter",
        eval_model: str = "gpt-4o"
):
    """Run test for a single model and case"""
    try:
        # Get model's maximum context length
        max_context = model_config['models'][model_name]['max_context']

        # If context_lengths is not specified, use default values
        if context_lengths is None:
            context_lengths = [3000, 5000, 8000, 12000, 16000, 24000, 30000, 48000, 64000,
                               96000, 127000, 156000, 192000, 256000, 386000, 512000, 786000, 999000]

        context_lengths = [l for l in context_lengths if l <= max_context]

        # If document_depth_percents is not specified, use default values
        if document_depth_percents is None:
            document_depth_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Get model provider
        model_to_test = get_model_provider(model_name, model_config)

        eval_model = "gpt-4o"

        # Get evaluation model's API key and base URL, throw error if not found
        eval_api_key = os.getenv(model_config['models'][eval_model]['api_key'])
        if eval_api_key is None:
            raise ValueError(f"API key for evaluation model {eval_model} not found")

        eval_base_url = os.getenv(model_config['models'][eval_model]['base_url'])
        if eval_base_url is None:
            raise ValueError(f"Base URL for evaluation model {eval_model} not found")

        # Get evaluator
        evaluator = None if only_context else OpenAIEvaluator(
            model_name=eval_model,
            question_asked=test_cases[case_name]["question"],
            true_answer=test_cases[case_name]["true_answer"],
            api_key=eval_api_key,
            base_url=eval_base_url
        )

        # Create multi-needle tester
        tester = LLMMultiNeedleHaystackTester(
            model_to_test=model_to_test,
            evaluator=evaluator,
            needles=test_cases[case_name]["needles"],
            retrieval_question=test_cases[case_name]["question"],
            haystack_dir=haystack_dir,
            context_lengths=context_lengths,
            document_depth_percents=document_depth_percents,
            document_depth_percent_interval_type=document_depth_percent_interval_type,
            num_concurrent_requests=num_concurrent_requests,
            save_results=save_results,
            save_contexts=save_contexts,
            final_context_length_buffer=final_context_length_buffer,
            print_ongoing_status=print_ongoing_status,
            case_name=case_name,
            only_context=only_context,
            enable_dynamic_sleep=enable_dynamic_sleep,
            base_sleep_time=base_sleep_time
        )

        # Start test
        tester.start_test()
        return True, f"Successfully completed test for {model_name} - {case_name}"
    except Exception as e:
        return False, f"Test for {model_name} - {case_name} failed: {str(e)}"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLM Multi-Needle Test Script')

    # Required parameters
    parser.add_argument('--model-names', nargs='+', required=True,
                        default="gpt-4o-mini",
                        help='List of model names to test, e.g.: gpt-4 claude-3')
    parser.add_argument('--case-names', nargs='+', required=True,
                        default="pizza_ingredients",
                        help='List of case names to test, e.g.: pizza_ingredients rainbow_potion')
    parser.add_argument('--eval-model', type=str, default="gpt-4o",
                        help='Evaluation model name, default is gpt-4o')

    # Optional parameters - context length and document depth
    parser.add_argument('--context-lengths', nargs='+', type=int,
                        default=[1000, 2000, 3000, 5000, 8000, 12000, 16000, 24000, 30000, 48000, 64000,
                                 96000, 127000, 156000, 192000, 256000, 386000, 512000, 786000, 999000],
                        help='Context length list, default is full range')
    parser.add_argument('--document-depth-percents', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='Document depth percentage list, default is full range')

    # Other optional parameters
    parser.add_argument('--num-concurrent-requests', type=int, default=1,
                        help='Number of concurrent requests, default is 1')
    parser.add_argument('--final-context-length-buffer', type=int, default=300,
                        help='Final context length buffer, default is 300')
    parser.add_argument('--base-sleep-time', type=float, default=0.5,
                        help='Base sleep time, default is 0.5 seconds')
    parser.add_argument('--haystack-dir', type=str, default='PaulGrahamEssays',
                        help='Data directory, default is PaulGrahamEssays')
    parser.add_argument('--depth-interval-type', type=str, default='linear',
                        choices=['linear'],
                        help='Document depth interval type, default is linear')

    # Boolean flags
    parser.add_argument('--no-save-results', action='store_false', dest='save_results',
                        help='Do not save results')
    parser.add_argument('--no-save-contexts', action='store_false', dest='save_contexts',
                        help='Do not save contexts')
    parser.add_argument('--no-print-status', action='store_false', dest='print_ongoing_status',
                        help='Do not print ongoing status')
    parser.add_argument('--only-context', action='store_true',
                        help='Only process context')
    parser.add_argument('--no-dynamic-sleep', action='store_false', dest='enable_dynamic_sleep',
                        help='Disable dynamic sleep')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    load_dotenv()

    # Load configuration files
    model_config = load_model_config('config/model_config.yaml')
    test_cases = load_test_cases('config/needle_cases.yaml')

    # Define test parameters
    test_params = {
        "context_lengths": args.context_lengths,
        "document_depth_percents": args.document_depth_percents,
        "document_depth_percent_interval_type": args.depth_interval_type,
        "num_concurrent_requests": args.num_concurrent_requests,
        "save_results": args.save_results,
        "save_contexts": args.save_contexts,
        "final_context_length_buffer": args.final_context_length_buffer,
        "print_ongoing_status": args.print_ongoing_status,
        "only_context": args.only_context,
        "enable_dynamic_sleep": args.enable_dynamic_sleep,
        "base_sleep_time": args.base_sleep_time,
        "haystack_dir": args.haystack_dir,
        "eval_model": args.eval_model
    }

    # Create all test combinations
    test_combinations = list(product(args.model_names, args.case_names))
    total_tests = len(test_combinations)

    # Create progress bar
    progress_bar = tqdm(total=total_tests, desc="Test Progress")
    progress_lock = threading.Lock()

    def update_progress(*args):
        with progress_lock:
            progress_bar.update(1)

    # Execute tests using thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for model_name, case_name in test_combinations:
            future = executor.submit(
                run_single_test,
                model_name,
                case_name,
                model_config,
                test_cases,
                **test_params
            )
            future.add_done_callback(update_progress)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            success, message = future.result()
            results.append(message)

    progress_bar.close()

    # Print test results summary
    print("\nTest Results Summary:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
