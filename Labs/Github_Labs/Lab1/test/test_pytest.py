import pytest
from src import calculator, word_count

def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5,0) == 5
    assert calculator.fun1 (-1, 1) == 0
    assert calculator.fun1 (-1, -1) == -2


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5,0) == 5
    assert calculator.fun2 (-1, 1) == -2
    assert calculator.fun2 (-1, -1) == 0

def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5,0) == 0
    assert calculator.fun3 (-1, 1) == -1
    
    assert calculator.fun3 (-1, -1) == 1

def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5,0, -1) == 4
    assert calculator.fun4 (-1, -1, -1) == -3
    
    assert calculator.fun4 (-1, -1, 100) == 98
    
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
