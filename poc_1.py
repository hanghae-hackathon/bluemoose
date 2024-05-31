from openai import OpenAI
import os
from langchain.schema import Document
from io import StringIO
from PyPDF2 import PdfReader


def extract_insurance_clauses(uploaded_files):

    print("extract_insurance_clauses method invoked!")

    # LLM에 질의
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # PDF 파일 열기
    # pdf_file = open(pdf_path, 'rb')
    # pdf_reader = PyPDF2.PdfReader(pdf_file)

    total_contents = ""

    if uploaded_files is not None:

        print('uploded_files length = ', len(uploaded_files))
        pdf_reader = ''
        for file in uploaded_files:

            print('file name = ', file.name)
            pdf_file = open(file.name, 'rb')
            pdf_reader = PdfReader(pdf_file)

        print('pdf file read finished!')
        print('pdf_reader.pages = ', pdf_reader.pages)
        text = ""
        # len(pdf_reader.pages)
        for page in [4,5,6]:

            print('page=  ', page)

            text = pdf_reader.pages[page].extract_text()
            prompt = f"""{text}\n
                위 텍스트는 보험 계약의 약관입니다. 다음 텍스트에서 보험 계약의 면책조항을 찾아주세요."""
            
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="gpt-4",
                temperature=0.2
            )

            content = response.choices[0].message.content

            if content is None or content[0] == 'N':
                continue

            total_contents += f"Page : {page+1} - " + content
    
    return parse_insurance_clauses(total_contents)

def parse_insurance_clauses(total_contents):

    print("parse_insurance_clauses method invoked!")

    # 결과 파싱
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response2 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": total_contents
            },
            {
                "role": "assistant", 
                "content": "[Page Number] 페이지 - [조항]\n"
            },
        ],
        model="gpt-4",
        temperature=0.2
    )

    responseText = response2.choices[0].message.content

    print("extract_insurance_clauses method finished!")

    return responseText.split("\n")