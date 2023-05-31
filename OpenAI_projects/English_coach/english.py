import semantic_kernel as sk
import asyncio
import os
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion, AzureTextCompletion

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
context = kernel.create_new_context()
context["history"] = ""
chat_function = kernel.create_semantic_function(prompt, "ChatBot", max_tokens=200, temperature=0.7, top_p=0.5)

async def chat() -> None:
    # Save new message in the context variables
    ask = input("Shengjie:")
    context["input"] = ask
    answer = await chat_function.invoke_async(context=context)
    print(f"Linda: {answer}")
    context["history"] += f"\nShengjie: {ask}\nLinda: {answer}"

while True:
    task = chat()
    asyncio.run(task)
