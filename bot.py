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

async def q(update: Update, _: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.replace('/q', '').strip() # type: ignore
    text = qa.run(query)
    await update.message.reply_text(text) # type: ignore

if __name__ == '__main__':
    token = os.environ['TELEGRAM_TOKEN']
    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler('q', q))
    application.run_polling()

