# pip install PyPDF2
import PyPDF2
from openai import OpenAI
import os

# LLM에 질의
client = OpenAI(
    # This is the default and can be omitted
    api_key= os.environ["OPENAI_API_KEY"]
)

# PDF 파일 열기
pdf_file = open("플러스메디컬단체보험_특별약관.pdf", 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)


for page in range(len(pdf_reader.pages)):
    text = pdf_reader.pages[page].extract_text()
# text = pdf_reader.pages[2].extract_text()

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

    # 응답 출력
    print(f"Page : {page} - {response.choices[0].message.content}")


pdf_file.close()
