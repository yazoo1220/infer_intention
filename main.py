import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def get_top_urls(keyword, k=2):
  '''
  Get top k urls from the search result of the keyword
  '''
  res = search.results(keyword)
  urls = list(map(lambda x: x['link'], res['organic_results'][:k]))
  return urls

def get_summary_by_url(url):
  loader = WebBaseLoader(url)
  docs = loader.load()
  # Split the document into texts
  texts = text_splitter.create_documents([str(docs)])

  # limit the length of the doc to read
  length = len(texts)
  if len(texts) > 3:
    length = 3

  summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
  summary = summary_chain.run(texts[:length])
  return summary

def infer_intention_from_summary(keyword, summary):
  prompt = ChatPromptTemplate.from_messages([(
      "system",
      """
      あなたはSEOコンサルタントです。
      """
      ), (
      "human",
      """
      この要約はある検索キーワードの上位の結果です。
      この内容が上位に来ているということからこの内容が検索意図を満足させるものと仮定できます。
      この内容から元の検索者の検索意図を推測してください。具体的には

　　　 意図の種類: Know Go Do Buy の四分類のどれに当てはまるか
　　　 属性:年齢層・法人か個人か・家族構成・人生の段階など）の人間か
      コンテンツタイプ:　記事　サービス　EC その他などに分類
      顕在ニーズ: 想定読者が既に持っていて記事を読む際に頭に置いているニーズ
      潜在ニーズ: 想定読者自身意識していないが、何かのきっかけで顕在化しうるニーズ
      シチュエーション: どんなシチュエーションで何を目的にして検索したかを推測してください。
      不満点: またこのサイトのどのような点に不満を感じるかも解説してください。

      元の検索キーワードは「{keyword}」です。
      回答は日本語でお願いします。ステップバイステップで考えよう。
      {input}
      """
      )
      ])
  int_chain = prompt | llm
  intention = int_chain.invoke({"keyword": keyword, "input": summary})
  return intention

def infer_intention_from_keyword(keyword, k=2):
  urls = get_top_urls(keyword, k)
  print('urls are fetched')
  summaries = []
  for url in urls:
    print('summarizing ' + url)
    summary = get_summary_by_url(url)
    summaries.append(summary)
  print(summaries)

  intentions = []
  for summary in summaries:
    intention = infer_intention_from_summary(keyword, summary)
    print(intention)
    intentions.append(intention)

  return intentions


# title
st.title('🔍 検索意図逆算ツール')
st.markdown('検索結果の上位の内容を要約し、検索者の意図を測るための要素を抽出します')

# Fetch URLs to analyze
search = SerpAPIWrapper()
query = st.text_input('検索キーワード', placeholder='マンション リノベーション')
top_k = st.slider('表示する結果', 1, 5, 2)

# Create an instance of the RecursiveCharacterTextSplitter
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

if query:
    query_button = st.button("実行")

    # session_stateにresponsesのリストが存在しない場合、新しいリストを作成
    if 'all_responses' not in st.session_state:
        st.session_state.all_responses = []

    if query_button or ('download_clicked' in st.session_state and st.session_state.download_clicked):
        with st.spinner("..."):
            if query_button:  # 新しいクエリが実行された場合のみresponsesを取得
                new_responses = infer_intention_from_keyword(query, top_k)
                new_responses = [res.content for res in new_responses]
                st.session_state.all_responses.extend(new_responses)

            # 全てのresponsesを表示
            content = "\n\n".join(st.session_state.all_responses)
            st.code(content)

            # ダウンロードボタンがクリックされたことをsession_stateでトラック
            if st.download_button("⬇️csv", content):
                st.session_state.download_clicked = True
            else:
                st.session_state.download_clicked = False



# def suggest_outline_from_intention(intention):
#   prompt = ChatPromptTemplate.from_messages([(
#       "system",
#       """
#       あなたはSEOコンサルタントです。
#       """
#       ), (
#       "human",
#       """
#       この要約はある検索キーワードにおける検索意図を分析したものです。
#       この内容を元に同じ検索意図を持った人が満足する記事の目次を３パターン考えてください。
#       記事の内容は下記の内容で限定します。
#       - リノベーション・リフォームに関すること
#       - マンションに関すること
#       回答は日本語でお願いします。水平思考で考えよう。

#       ###
#       検索意図：
#       {intention}
#       """
#       )
#       ])
#   outline_chain = prompt | llm
#   outline = outline_chain.invoke({"intention": intention})
#   return outline

# st.markdown(suggest_outline_from_intention(str(mansion_res[0])))
