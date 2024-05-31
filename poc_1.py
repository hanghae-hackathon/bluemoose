from openai import OpenAI
import os


def extract_insurance_clauses(pdf):

    # LLM에 질의
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # PDF 파일 열기
    # pdf_file = open(pdf_path, 'rb')
    # pdf_reader = PyPDF2.PdfReader(pdf_file)

    total_contents = ""

    # 조항 탐색
    for page in range(len(pdf.pages)):
        text = pdf.pages[page].extract_text()

        prompt = f"""{text}\n
            위 텍스트는 보험 계약의 약관입니다. 면책조항이 있으면 Y, 없으면 N으로 답해주세요.\n
            있을 경우에는 몇 조의 몇 항인지만 적어주세요."""
        
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

    return responseText.split("\n")