import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from poc_1 import extract_insurance_clauses
from RAG import configure_qa_chain, DocumentLoader
import pandas as pd
from voting import Voting, VECTOR_DIR

# openai settings 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

# initialize Voting instance
voting = Voting(VECTOR_DIR)


if __name__ == '__main__': 

    # UI 초기화
    #st.set_page_config(page_title="Bluemoose:Chat with Documents", page_icon="B")
    st.title('블루무스')
    
    uploaded_files = st.sidebar.file_uploader(
        label="파일 업로드",
        type=list(DocumentLoader.supported_extensions.keys()),
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("계속 진행하려면 도큐먼트를 업로드하세요.")
        st.stop()

    qa_chain = configure_qa_chain(uploaded_files)
    assistant = st.chat_message("assistant")
    user_query = st.chat_input(placeholder="무엇이든 물어보세요.")

    # user query 가 존재할 경우
    if user_query:

        stream_handler = StreamlitCallbackHandler(assistant)
        response_list = []

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            model="gpt-4-turbo",
        )
        
        if response.choices[0].message.content == 'A':
            response_list.append(extract_insurance_clauses(uploaded_files))

            joined_response_list = []
            for response in response_list:
                joined_response_list.append(' '.join(response))

            verification_list = []
            for joined_response in joined_response_list:
                verification_list.append(voting.voting(joined_response))

            # Initialize data for the DataFrame
            data = {
                "면책조항": joined_response_list,  # Sample data for column "면책조항"
                "판단": verification_list      # Sample data for column "검증"
            }

            # Create DataFrame
            df = pd.DataFrame(data)
            st.table(df)

            # Inject CSS to adjust the width of the second column
            st.markdown(
                """
                <style>
                table th:nth-child(3) {
                    width: 50px; 
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        else: 
            response = qa_chain.run(user_query, callbacks=[stream_handler])
            print(response)
            st.markdown(response)

    print('============================FINISHED============================')

        



