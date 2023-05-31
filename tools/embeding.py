import os
import pandas as pd
import tiktoken
import openai



openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# Newest Version
openai.api_version = "2023-03-15-preview"
AZURE_DEPLOYMENT_NAME='textdavinci003'
max_tokens=500

def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie
# Function to split the text into chunks of a maximum number of tokens
def split_into_many(tokenizer,text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

def embeding_from_txt(file_name:str,embeding_path:str) -> None:
    # Create a list to store the text files
    texts = []
    with open(file_name, 'r',encoding='utf-8') as f:
        text = f.read()
        texts.append(("fname", text))
    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['fname', 'text'])
    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(embeding_path,encoding='utf-8')
    df.head()
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(embeding_path, index_col=0,encoding='utf-8')
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()

    shortened = []
    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(tokenizer,row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv(embeding_path,encoding='utf-8')
    df.head()
if __name__ == '__main__':
    filepath="./files/Gartner_wms_2023.txt"
    embeding_path='./files/embeddings.csv'
    embeding_from_txt(filepath,embeding_path)

