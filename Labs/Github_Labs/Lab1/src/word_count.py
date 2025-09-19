def count_words(filename):
    try:
        with open(filename, "r") as file:
            text = file.read()

            lines = text.split("\n")
            words = text.split()
            characters = list(text)

            print("Lines:", len(lines))
            print("Words:", len(words))
            print("Characters:", len(characters))

    except FileNotFoundError:
        print("Error: File not found.")


if __name__ == "__main__":
    filename = input("Enter the filename: ")
    count_words(filename)
