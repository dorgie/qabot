#!/usr/bin/env python

import datetime
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, PicklePersistence

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from index import MyKNNRetriever

retriever = MyKNNRetriever.from_file('index.pkl')
llm = ChatOpenAI(temperature=0, client=None)
llm = OpenAI(client=None)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True, return_source_documents=False)

def log(update: Update):
    m = update.message # type: ignore
    now = datetime.datetime.now()
    print(f'{now} [{m.chat_id}] {m.from_user.first_name} {m.from_user.last_name} {m.from_user.username}: {m.text}') # type: ignore

async def auth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user # type: ignore
    authorized_users = context.bot_data.get('authorized_users', []) # type: ignore
    if user and user.id in authorized_users:
        return True
    await update.message.reply_text('Waiting for authorization') # type: ignore
    admin = int(os.environ['ADMIN_CHAT'])
    text = f'User {user.id} {user.first_name} {user.last_name} {user.username} wants to use the bot\n/approve {user.id}' # type: ignore
    await context.bot.send_message(chat_id=admin, text=text) # type: ignore
    return False

async def q(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log(update)
    if not await auth(update, context):
        return
    query = update.message.text.replace('/q', '').strip() # type: ignore
    if 'index' not in context.chat_data: # type: ignore
        chain = qa
    else:
        index = context.chat_data['index'] # type: ignore
        r = MyKNNRetriever(**index)
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=r, verbose=True, return_source_documents=False)
    text = chain.run(query) # type: ignore
    await update.message.reply_text(text) # type: ignore

async def index_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log(update)
    if not await auth(update, context):
        return
    url = update.message.text.replace('/index', '').strip() # type: ignore
    r = MyKNNRetriever.from_url(url)
    context.chat_data['index'] = {'index': r.index, 'texts': r.texts} # type: ignore
    #context.chat_data['qa'] = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=r, verbose=True, return_source_documents=False) # type: ignore
    text = f'indexed {len(r.texts)} documents'
    await update.message.reply_text(text) # type: ignore

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log(update)
    if update.message.chat.id != int(os.environ['ADMIN_CHAT']): # type: ignore
        await update.message.reply_text('Permission denied') # type: ignore
        return
    user = update.message.text.replace('/approve', '').strip() # type: ignore
    authorized_users = context.bot_data.get('authorized_users', []) # type: ignore
    authorized_users.append(int(user))
    context.bot_data['authorized_users'] = authorized_users # type: ignore
    await update.message.reply_text(f'User {user} approved') # type: ignore
    await context.bot.send_message(chat_id=int(user), text='You are approved') # type: ignore

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log(update)
    if not await auth(update, context):
        return

if __name__ == '__main__':
    token = os.environ['TELEGRAM_TOKEN']
    persistence = PicklePersistence(filepath="context.pkl")
    application = ApplicationBuilder().token(token).persistence(persistence).build()
    application.add_handler(CommandHandler('q', q))
    application.add_handler(CommandHandler('index', index_cmd))
    application.add_handler(CommandHandler('approve', approve))
    application.add_handler(CommandHandler('start', start))
    application.run_polling()

