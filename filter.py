import re

# List of inappropriate or unsafe words
INAPPROPRIATE_WORDS = ['kötü', 'şiddet', 'tehlike', 'ölüm', 'yaralanma']

def check_for_inappropriate_content(text):
    """
    Checks for inappropriate content in the given text.
    Returns a child-friendly warning if inappropriate content is found.
    """
    for word in INAPPROPRIATE_WORDS:
        if re.search(fr'\b{word}\b', text, re.IGNORECASE):
            return "Üzgünüm, bu çok ciddi bir şey! Daha eğlenceli ve güvenli bir cümle yazalım mı? 😊"
    return text  # Return original text if clean

if __name__ == "__main__":
    # Test inappropriate content filter
    test_input = "Bu cümlede tehlike var."
    print("Input:", test_input)
    print("Filter Result:", check_for_inappropriate_content(test_input))
