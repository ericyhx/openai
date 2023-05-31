import pandas as pd
import tiktoken
import openai
import numpy as np
import os

from openai.embeddings_utils import distances_from_embeddings

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# Newest Version
openai.api_version = "2023-03-15-preview"
AZURE_DEPLOYMENT_NAME='textdavinci003'


embeding_path= './files/embeddings.csv'
df=pd.read_csv(embeding_path, index_col=0,encoding='utf-8')
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()
tokenizer = tiktoken.get_encoding("cl100k_base")

def create_context(
    question, df, max_len=500, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="textdavinci003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=500,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        global token_prompt
        token_prompt =  len(tokenizer.encode(prompt))
        print ('\n Token_prompt:'+ str(token_prompt))
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=prompt,
            #prompt=f"Answer the question in English. Answer the question based on the context below, \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0.1,
            max_tokens=max_tokens,
            top_p= 0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            engine="textdavinci003",
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

Prompt = """
    Read the Gartner report. Give me the information about Körber.

    Here's an example. Please provide the information the same as the example below

    Partner name: Blue Yonder
    2022 WMS software Revenue: 190 MILLION
    WMS customer amount: 1000
    Industry: 3PL,food and beverage, cusumer products, retail and pharmaceuticals
    Customer base: 50% in North America
    Partner：total 88, 39 in North America, 17 in Europe, 20 in Asia, 12 in Latin America
    In which level WMS are often used: 3,4
    Product portfolio: Adaptive Fulfillment and Warehousing
    The progress of SaaS: 80% customers on SaaS and 90% of new bookings now SaaS
"""
Completion = answer_question(df, question=Prompt, debug=False)
token_completion = len(tokenizer.encode(Completion))
token_total = token_completion + token_prompt
print ('\n Token_completion:'+ str(token_completion))
print ('\n Token_total:'+ str(token_total) + '\nrmb:' + str(0.02/1000*token_total*0.68))
print('\nGeek+ Assistant:\n' + Completion + '\n')
