import torch

'''
2. Select a relatively large file containing Python source code. 
a) Build an index of all the words in the source file (
feel free to make your tokenization as simple or as complex as you like; we suggest starting with replacing r"[
^a-zA-Z0-9_]+" with spaces). 
b) Compare your index with the one we made for Pride and Prejudice . Which is larger? 
c) Create the one-hot encoding for the source code file. 
d) What information is lost with this encoding? How does that 
information compare to what’s lost in the Pride and Prejudice encoding?
'''


def process_words(directory: str, file_name: str):
    # Load the .txt file
    file_path = directory + file_name
    with open(file_path, encoding='utf8') as f:
        text = f.read()

    # Split the text into a list of cleaner words
    words = split_clean_words(text=text)

    # load index into dictionary that indexes the words
    words_index = index_words(words)
    print(f"\nWord index: {words_index['he']}")
    print("ending...")

    # Create one-hot encoding of the words
    words_one_hot = one_hot_encode_words(words_index=words_index, words=words)

    # Playing around with adding the words with concatenations
    # Concat the one-hot-encoding of words
    # word_data = torch.arange(0, len(words)).unsqueeze(1)
    # word_data = torch.cat((word_data, words_one_hot), dim=1)

    # Create a numpy array with the word strings
    # word_data_np = word_data.numpy()
    # words_np = np.array(words)[:, None]
    # # words_np = words_np[:, None]
    # word_data_np = np.concatenate((words_np, word_data_np), axis=1)
    # return word_data, word_data_np

    return words_one_hot


def one_hot_encode_words(words_index: dict, words: list) -> torch.tensor:
    words_hot = torch.zeros(len(words), len(words_index))

    # One-hot-encode each word
    for i, word in enumerate(words):
        words_hot[i][words_index[word]] = 1
    sum = words_hot.sum()
    return words_hot


def split_clean_words(text: str) -> list:
    # split the words to a list
    words = text.lower().replace('\n', ' ').split()

    # strip_chars = "[^a-zA-Z0-9_]+.,;:!?/”/“_-"
    strip_chars = '.,;:"!?”“_-&($*, '
    words = [word.strip(strip_chars) for word in words]

    # Return the list sorted with a unique set
    return sorted(set(words))


def index_words(words: list) -> dict:
    word_index = {word: i for (i, word) in enumerate(words)}
    return word_index
