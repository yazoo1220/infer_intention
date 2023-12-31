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

@st.cache(suppress_st_warning=True, show_spinner=False)
def get_top_urls(keyword, k=2):
    '''
    Get top k urls and titles from the search result of the keyword
    '''
    res = search.results(keyword)
    results = [{'link': r['link'], 'title': r['title']} for r in res['organic_results'][:k]]
    return results

@st.cache(suppress_st_warning=True, show_spinner=False)
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

@st.cache(suppress_st_warning=True, show_spinner=False)
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
    results = get_top_urls(keyword, k)
    print('URLs and titles are fetched')
    summaries = []
    for result in results:
        print('summarizing ' + result['link'])
        summary = get_summary_by_url(result['link'])
        summaries.append({'url': result['link'], 'title': result['title'], 'summary': summary})
    print(summaries)

    intentions = []
    for summary in summaries:
        intention = infer_intention_from_summary(keyword, summary['summary'])
        print(intention)
        intentions.append({'url': summary['url'], 'title': summary['title'], 'intention': intention.content})

    return intentions

def synthesize_summary(responses):
    """responsesの内容を総合的に要約する関数"""
    combined_text = "\n".join(responses)
    synthesis_chain = load_summarize_chain(llm, chain_type="map_reduce")
    synthesized_summary = synthesis_chain.run([combined_text])
    return synthesized_summary[0]


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

# ...

# ...

if query:
    query_button = st.button("実行")

    if query_button or ('download_clicked' in st.session_state and st.session_state.download_clicked):
        with st.spinner("..."):
            # クエリが実行された場合のみ新しいレスポンスを取得
            if query_button:
                new_responses = infer_intention_from_keyword(query, top_k)
                formatted_responses = []
                for res in new_responses:
                    formatted_responses.append(f"URL: {res['url']}\nTitle: {res['title']}\n\n{res['intention']}")
                st.session_state.all_responses = formatted_responses
                

            
    # 全てのresponsesを表示
    if 'all_responses' in st.session_state:
        all_content = "\n\n".join(st.session_state.all_responses)
        st.code(all_content)
        
        if st.download_button("すべてのレスポンスをダウンロード⬇️csv", all_content):
            st.session_state.download_clicked = True
        else:
            st.session_state.download_clicked = False
            
        summarize_button = st.button("さらに要約")
        if summarize_button:
            overall_summary = create_overall_summary(st.session_state.all_responses)
            st.session_state.overall_summary = overall_summary
            st.subheader("総合的な要約")
            st.code(overall_summary)
            if st.download_button("⬇️", overall_summary):
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
