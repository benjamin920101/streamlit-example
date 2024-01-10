# Importing necessary packages, files and services
import os

import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = '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'

# App UI framework
st.title('🦜🔗 News Generator')
prompt = st.text_input('News topic: ')

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='請寫一則關於 {topic} 的新聞'
)

tweet_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template='你是一個醫療保健的公共衛生新聞專家，請依循我的關鍵字，產製一篇新聞的引言與相關介紹。主題： {title} ，以下是關於 {title} 的維基百科資料：{wikipedia_research} '
)

final_title_template = PromptTemplate(
    input_variables = ['tweet'],
    template='根據以上生成的新聞，請為此新聞撰寫一個標題：{tweet}'
)

# Wikipedia data
wiki = WikipediaAPIWrapper(
    lang="zh-tw",
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
tweet_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
final_title_memory = ConversationBufferMemory(input_key='tweet', memory_key='chat_history')

# Llms
llm = OpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.5)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
tweet_chain = LLMChain(llm=llm, prompt=tweet_template, verbose=True, output_key='script', memory=tweet_memory)
final_title_chain = LLMChain(llm=llm, prompt=final_title_template, verbose=True, output_key='final', memory=final_title_memory)

# Chaining the components and displaying outputs
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    tweet = tweet_chain.run(title=title, wikipedia_research=wiki_research)
    final_title_chain = final_title_chain.run(tweet=tweet)
	 
    st.write(final_title_chain)
    st.write(tweet)
 

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('News History'):
        st.info(tweet_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
