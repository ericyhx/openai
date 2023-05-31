import semantic_kernel as sk
import asyncio
import os
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion, AzureTextCompletion
import streamlit as st
from streamlit_chat import message
import requests

kernel = sk.Kernel()


# Prepare OpenAI service using credentials stored in the `.env` file
#api_key, org_id = sk.openai_settings_from_dot_env()
#kernel.add_text_completion_service("dv", OpenAITextCompletion("text-davinci-003", api_key, org_id))

# Alternative using Azure:
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", AzureTextCompletion(deployment, endpoint, api_key))

# note: using skills from the samples folder
skills_directory = "./"
LearningFunctions = kernel.import_semantic_skill_from_directory(skills_directory, "Learning")
LearningFunction = LearningFunctions["English"]

# TEST 1
###################################################

#history = ""
#prompt = ""
#while prompt!= "quit":
#    ask = input("What's you question: ")
#    result = LearningFunction(ask)
#    history += ask+"\n"
#    history += result.__str__()+"\n"
#    print (history)


# TEST 2
###################################################

prompt = ""
with open ("./Learning/English/skprompt.txt", 'r') as f:
    prompt = f.read()
#context = kernel.create_new_context()
#context["history"] = ""
chat_function = kernel.create_semantic_function(prompt, "ChatBot", max_tokens=200, temperature=0.7, top_p=0.5)

#while True:
#    task = chat()
#    asyncio.run(task)

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

#API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
#headers = {"Authorization": st.secrets['api_key']}

st.header("Streamlit Chat - Demo")
st.markdown("[Github](https://github.com/ai-yash/st-chat)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
def get_text():
    input_text = st.text_area("You: ", key="input")
    #print (dir(input_text))
    return input_text


if 'context' not in st.session_state:
    st.session_state['context'] = kernel.create_new_context()
    st.session_state['context']["history"] = ""

async def chat() -> None:
    user_input = get_text()
    print ("START #################")
    if user_input:
        st.session_state['context']["input"] = user_input
        output = await chat_function.invoke_async(context=st.session_state['context'])
        output_string = output.__str__()

        st.session_state['context']["history"] += f"\nShengjie: {user_input}\nLinda: {output}"
        print ("##### Histroy start ####")
        print (st.session_state['context']["history"])
        print ("##### Histroy end ####")

        #output = "HIIII"
        #output = query({
        #    "inputs": {
        #        "past_user_inputs": st.session_state.past,
        #        "generated_responses": st.session_state.generated,
        #        "text": user_input,
        #    },"parameters": {"repetition_penalty": 1.33},
        #})

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output_string)

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


task = chat()
asyncio.run(task)
