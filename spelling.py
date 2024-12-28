import re
from typing import Dict, List, Tuple

class TurkishSpellingCorrector:
    def __init__(self):
        self.TURKISH_CORRECTIONS: Dict[str, str] = {
            "küçuk": "küçük",
            "sarı": "sarı",
            "saclı": "saçlı",
            "cocuk": "çocuk",
            "izlior": "izliyor",
            "gelior": "geliyor",
            "bakıor": "bakıyor",
            "gidior": "gidiyor",
            "agac": "ağaç",
            "kardas": "kardeş",
            "yagmur": "yağmur",
            "ogretmen": "öğretmen",
            "ogrenci": "öğrenci"
        }

        self.CHAR_REPLACEMENTS: Dict[str, str] = {
            'g': 'ğ',
            'i': 'ı',
            'o': 'ö',
            'u': 'ü',
            's': 'ş',
            'c': 'ç'
        }

    def correct_spelling(self, text: str) -> str:
        words = text.split()
        corrected_words = []

        for word in words:
            if word.lower() in self.TURKISH_CORRECTIONS:
                corrected = self.TURKISH_CORRECTIONS[word.lower()]
                if word.istitle():
                    corrected = corrected.capitalize()
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def suggest_corrections(self, word: str) -> List[str]:
        suggestions = []

        if word.lower() in self.TURKISH_CORRECTIONS:
            suggestions.append(self.TURKISH_CORRECTIONS[word.lower()])

        for char, replacement in self.CHAR_REPLACEMENTS.items():
            if char in word:
                suggested_word = word.replace(char, replacement)
                if suggested_word != word:
                    suggestions.append(suggested_word)

        return list(set(suggestions))

    def check_sentence(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        words = text.split()
        corrections = []
        corrected_words = []

        for word in words:
            if word.lower() in self.TURKISH_CORRECTIONS:
                correction = {
                    'original': word,
                    'corrected': self.TURKISH_CORRECTIONS[word.lower()],
                    'suggestions': self.suggest_corrections(word)
                }
                corrections.append(correction)
                corrected_words.append(self.TURKISH_CORRECTIONS[word.lower()])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), corrections

def main():
    # Initialize the spelling corrector
    corrector = TurkishSpellingCorrector()

    # Test cases
    test_sentences = [
        "küçuk sarı saclı bir cocuk yıldızları izlior.",
        "Agac altında ogrenci kitap okior.",
        "Kardas ile beraber yagmur altında yürüyoruz."
    ]

    print("Testing Turkish Spelling Corrector:")
    print("-" * 50)

    for sentence in test_sentences:
        print("\nOriginal:", sentence)
        
        # Get corrections and suggestions
        corrected, corrections = corrector.check_sentence(sentence)
        
        print("Corrected:", corrected)
        
        # Print detailed corrections
        if corrections:
            print("\nDetailed corrections:")
            for correction in corrections:
                print(f"- {correction['original']} → {correction['corrected']}")
                if correction['suggestions']:
                    print(f"  Other suggestions: {', '.join(correction['suggestions'])}")

if __name__ == "__main__":
    main()
