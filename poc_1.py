# pip install PyPDF2
import PyPDF2
from openai import OpenAI

# LLM에 질의
client = OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
)

# PDF 파일 열기
pdf_file = open("플러스메디컬단체보험_특별약관.pdf", 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

for page in range(len(pdf_reader.pages)):
    text = pdf_reader.pages[page].extract_text()
    prompt = f"다음 텍스트에서 보험 계약의 면책조항을 찾아주세요:\n\n{text}"

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-4o",
    )

    # 응답 출력
    print(f"@.@ page: {page} - {response.choices[0].message.content}")

# # response = client.chat.completions.create(
# #     engine="text-davinci-003",
# #     prompt=prompt,
# #     max_tokens=1024,
# #     n=1,
# #     stop=None,
# #     temperature=0.5,
# # )

pdf_file.close()
