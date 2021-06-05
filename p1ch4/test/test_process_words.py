from p1ch4.process_words import process_words


def test_process_words():
    print("Running process_words() for Exercise 2")
    words_one_hot = process_words(directory="../../data/p1ch4/my-text/", file_name="paddy_at_home.txt")
    print(f"word one-hot-encoding shape: {words_one_hot.shape}")
