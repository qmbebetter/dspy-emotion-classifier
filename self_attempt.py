import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy
from dotenv import load_dotenv


def load_go_emotion_dataset() -> dict:
    """
    Loads the Go Emotions dataset into train, validation, and test splits.

    Returns:
        dict: Dataset splits with keys 'train', 'validation', and 'test'.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        return load_dataset("google-research-datasets/go_emotions", trust_remote_code=True)


def prepare_go_emotions_dataset(data_split, start: int, end: int) -> List[dspy.Example]:
    """
    Prepares a sliced dataset split of Go Emotions for use with DSPy.

    Args:
        data_split: The dataset split (e.g., train or test).
        start (int): Starting index of the slice.
        end (int): Ending index of the slice.

    Returns:
        List[dspy.Example]: List of DSPy Examples with text and expected labels (as emotion strings).
    """
    index_to_emotion = {
        0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
        5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
        10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear',
        15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
        20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse',
        25: 'sadness', 26: 'surprise', 27: 'neutral'
    }

    def indices_to_labels(indices):
        return [index_to_emotion[idx] for idx in indices]

    return [
        dspy.Example(
            text=row["text"],  # Input text from the dataset
            expected_labels=indices_to_labels(row["labels"])  # Convert indices to labels
        ).with_inputs("text")  # Specify the input field for DSPy
        for row in data_split.select(range(start, end))
    ]


# Load the dataset
dataset = load_go_emotion_dataset()

# Prepare the training and test sets
train_set = prepare_go_emotions_dataset(dataset["train"], 0, 50)
test_set = prepare_go_emotions_dataset(dataset["test"], 0, 50)


class EmotionClassification(dspy.Signature):
    """
    Classify the emotion(s) expressed in a given text.
    Output one or more emotion labels corresponding to predefined emotion classes.
    """
    text: str = dspy.InputField(desc="Input text to classify emotions from")
    extracted_emotions: List[str] = dspy.OutputField(
        desc="""Classify the emotion(s) in the following text. Choose from the predefined emotion labels:
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'.

        The output should be a JSON-formatted list of 1 to 5 selected emotions. Example: ["joy", "surprise"]."""
    )


# Create a ChainOfThought instance
emotion_classifier = dspy.ChainOfThought(EmotionClassification)

# Load .env file and OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure the language model
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)


def emotion_classification_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Computes correctness of emotion classification predictions by comparing predicted labels
    with the ground truth labels using Jaccard similarity.

    Args:
        example (dspy.Example): The dataset example containing expected emotion labels.
        prediction (dspy.Prediction): The prediction from the DSPy emotion classification program.
        trace: Optional trace object for debugging.

    Returns:
        float: Jaccard similarity score between predicted and actual labels.
    """
    predicted_emotions = set(prediction.extracted_emotions)
    actual_emotions = set(example.expected_labels)

    # Compute Jaccard similarity
    intersection = len(predicted_emotions & actual_emotions)
    union = len(predicted_emotions | actual_emotions)
    jaccard_similarity = intersection / union if union > 0 else 0.0

    print(f"Predicted: {predicted_emotions}, Actual: {actual_emotions}, Jaccard: {jaccard_similarity}")
    return jaccard_similarity


# Create an evaluator
evaluate_correctness = dspy.Evaluate(
    devset=train_set,
    metric=emotion_classification_metric,
    num_threads=6,
    display_progress=True,
    display_table=True
)

# Optimize the classifier
mipro_optimizer = dspy.MIPROv2(
    metric=emotion_classification_metric,
    auto="medium"
)

optimized_emotion_classifier = mipro_optimizer.compile(
    emotion_classifier,
    trainset=train_set,
    num_trials=5,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    requires_permission_to_run=False,
    minibatch=False,
)

# Evaluate the optimized classifier
evaluate_correctness(optimized_emotion_classifier, devset=train_set)