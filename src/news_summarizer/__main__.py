from .pipeline import analyze_text


if __name__ == "__main__":
    text = input("Paste article text: ")
    result = analyze_text(text)
    print(result.summary)
