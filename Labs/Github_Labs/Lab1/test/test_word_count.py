from src import word_count

def test_count_words_in_text_simple():
    text = "Hello world\nPython is fun"
    result = word_count.count_words_in_text(text)
    assert result["lines"] == 2
    assert result["words"] == 5
    assert result["characters"] == len(text)

def test_count_words_file(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello\nWorld")

    result = word_count.count_words(str(test_file))
    assert result["lines"] == 2
    assert result["words"] == 2
    assert result["characters"] == 11
