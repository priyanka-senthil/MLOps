
def count_words_in_text(text: str) -> dict:
    """Return counts of lines, words, and characters in given text."""
    lines = text.split("\n")
    words = text.split()
    characters = list(text)

    return {
        "lines": len(lines),
        "words": len(words),
        "characters": len(characters)
    }


def count_words(filename: str) -> dict:
    """Read a file and return counts of lines, words, and characters."""
    try:
        with open(filename, "r") as file:
            text = file.read()
            return count_words_in_text(text)
    except FileNotFoundError:
        return {"error": "File not found"}


if __name__ == "__main__":
    filename = input("Enter the filename: ")
    result = count_words(filename)
    if "error" in result:
        print(result["error"])
    else:
        print("Lines:", result["lines"])
        print("Words:", result["words"])
        print("Characters:", result["characters"])
