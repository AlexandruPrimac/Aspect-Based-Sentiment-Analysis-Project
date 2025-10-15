# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints

# Load the model directly from Hugging Face Hub
aspect_extractor = ATEPC.AspectExtractor(
    'multilingual',          # Can be replaced with a specific checkpoint name or a local file path
    auto_device=True,        # Use GPU/CPU or Auto
    cal_perplexity=True      # Calculate text perplexity
)
texts = [
    "这家餐厅的牛排很好吃，但是服务很慢。",
    "The battery life is terrible but the camera is excellent."
]
# Perform end-to-end aspect-based sentiment analysis
result = aspect_extractor.predict(
    texts,
    print_result=True,       # Console Printing
    save_result=False,       # Save results into a json file
    ignore_error=True,       # Exception handling for error cases
    pred_sentiment=True      # Predict sentiment for extracted aspects
)

# The output automatically identifies aspects and their corresponding sentiments:
# {
#   "text": "The user interface is brilliant, but the documentation is a total mess.",
#   "aspect": ["user interface", "documentation"],
#   "position": [[4, 19], [41, 54]],
#   "sentiment": ["Positive", "Negative"],
#   "probability": [[1e-05, 0.0001, 0.9998], [0.9998, 0.0001, 1e-05]],
#   "confidence": [0.9997, 0.9997]
# }