#!/usr/bin/env python

import os

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from index import MyKNNRetriever

retriever = MyKNNRetriever.from_file('index.pkl')
llm = OpenAI(client=None)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True, return_source_documents=False)

async def q(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.replace('/q', '').strip() # type: ignore
    text = context.chat_data.get('qa', qa).run(query) # type: ignore
    await update.message.reply_text(text) # type: ignore

async def index_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.replace('/index', '').strip() # type: ignore
    r = MyKNNRetriever.from_url(url)
    context.chat_data['qa'] = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=r, verbose=True, return_source_documents=False) # type: ignore
    text = f'indexed {len(r.texts)} documents'
    await update.message.reply_text(text) # type: ignore

if __name__ == '__main__':
    token = os.environ['TELEGRAM_TOKEN']
    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler('q', q))
    application.add_handler(CommandHandler('index', index_cmd))
    application.run_polling()

