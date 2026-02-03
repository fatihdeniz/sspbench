import utils
from evaluator import Evaluator

LABEL_SENTIMENT = {
    0: "negative",
    1: "positive",
    -1: "neutral"
}

class DecodingTrustEvaluator(Evaluator):
    
    def __init__(self, eval_model=None):
        super().__init__(eval_model)
    
    def _evaluate(self, instance, **kwargs):
        response = instance.get('response', '')
        sentiment_score = utils.parse_sentiment(response)
        instance['sentiment_score'] = sentiment_score
        instance['sentiment_label'] = LABEL_SENTIMENT.get(sentiment_score, "unknown")
        return sentiment_score
