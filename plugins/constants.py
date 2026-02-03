from enum import Enum

# Enum class for code security tests, string identifiers are the test names from the api project.
class CodeSecurityTests(str, Enum):
    INSTRUCT = "cybersec-eval-3-instruct"
    AUTOCOMPLETE = "cybersec-eval-3-autocomplete"
    SECCODEPLT_INSTRUCT = "seccodeplt-instruct"
    SECCODEPLT_AUTOCOMPLETE = "seccodeplt-autocomplete"
    
# Enum class for model and data privacy tests, string identifiers are the test names from the api project.
class PrivacyTests(str, Enum) :
    TRUSTLLM = "privacy-awareness-trustllm"
    CONFAIDE = "privacy-awareness-confaide" # Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, and Yejin Choi. Can llms keep a secret? testing privacy implications of language models via contextual integrity theory, 2023.
    ENRON = "privacy-leakage-enron"
    ECHR = "privacy-leakage-echr" 