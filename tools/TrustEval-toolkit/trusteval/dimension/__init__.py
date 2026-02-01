"""TrustAGI Evaluation Dimensions"""

# Ethics
from .ethics.ethics_llm.pipeline import pipeline as ethics_llm
from .ethics.ethics_vlm.pipeline import pipeline as ethics_vlm

# Fairness
# from .fairness.fairness_vlm.pipeline import run as fairness_vlm
# # from .fairness.fairness_llm.pipeline import pipeline as fairness_llm
# # from .fairness.fairness_t2i.pipeline import pipeline as fairness_t2i

# # Safety 
# from .safety.safety_vlm.pipeline import pipeline as safety_vlm
# from .safety.safety_llm.pipeline import pipeline as safety_llm
# from .safety.safety_t2i.pipeline import pipeline as safety_t2i

# # Robustness
# from .robustness.robustness_vlm.pipeline import pipeline as robustness_vlm  
# from .robustness.robustness_llm.pipeline import pipeline as robustness_llm
# from .robustness.robustness_t2i.pipeline import pipeline as robustness_t2i

# # Privacy
# from .privacy.privacy_t2i import pipeline as privacy_t2i
# # from .privacy.privacy_llm import pipeline as privacy_llm
# # from .privacy.privacy_vlm import pipeline as privacy_vlm

# # Truthfulness
# from .truthfulness.truthfulness_llm import pipeline as truthfulness_llm
# # from .truthfulness.truthfulness_vlm import pipeline as truthfulness_vlm
# from .truthfulness.truthfulness_t2i import pipeline as truthfulness_t2i

# # AI Risk
# from .ai_risk import dynamic_dataset_generator


## all t2i
# from trusteval.dimension.truthfulness.truthfulness_t2i import dynamic_dataset_generator no!
#from trusteval.dimensioni import dynamic_dataset_generator yes!
# how to make yes!?? just make it short not function
# from trusteval.dimension.truthfulness.truthfulness_t2i import dynamic_dataset_generator

from .truthfulness.truthfulness_t2i import pipeline as truthfulness_t2i


__all__ = [
    'truthfulness_t2i',
    'safety_llm',
    'ethics_llm',
    'ethics_vlm',
    # 'fairness_vlm',
    # 'fairness_llm',
    # 'fairness_t2i',
]