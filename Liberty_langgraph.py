# 필요한 패키지 설치
# !pip install transformers langchain torch faiss-cpu

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate
from langchain.chains import SequentialChain
from langchain.graphs import ChainGraph
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# # 1. 법률 관련 데이터를 파인 튜닝한 riberty-law-koBERT 모델 로드
# bert_tokenizer = BertTokenizer.from_pretrained('riberty-law-koBERT')
# bert_model = BertForSequenceClassification.from_pretrained('riberty-law-koBERT')

bert_tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertForSequenceClassification.from_pretrained("monologg/kobert")

# 2. GPT-4o 모델 및 토크나이저 로드 (예시로 GPT-2 사용)
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt-4o-mini-2024-07-18')
gpt_model = AutoModelForCausalLM.from_pretrained('gpt-4o-mini-2024-07-18')

# GPT-4o를 LangChain의 LLM으로 래핑
from transformers import pipeline
gpt_pipeline = pipeline('text-generation', model=gpt_model, tokenizer=gpt_tokenizer)
gpt_llm = HuggingFacePipeline(pipeline=gpt_pipeline)

# 3. 임베딩 및 벡터스토어 설정 (RAG용)
embeddings = HuggingFaceEmbeddings("nvidia/NV-Embed-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# 예시 문서 데이터 (법률 관련 예시 데이터)
documents = [
    "계약은 양 당사자 간의 합의에 의해 성립됩니다.",
    "계약의 위반 시 손해배상이 청구될 수 있습니다.",
    "계약서에 명시된 조항은 법적 구속력을 가집니다."
]

# 문서 분할 및 벡터스토어 생성       n
texts = []
for doc in documents:
    texts.extend(text_splitter.split_text(doc))

vectorstore = FAISS.from_texts(texts, embeddings)

# 4. koBERT 평가 함수 (법률 관련 질문 평가)
def evaluate_question_with_kobert(question):
    inputs = bert_tokenizer(question, return_tensors='pt')
    outputs = bert_model(**inputs)
    classification = torch.argmax(outputs.logits, dim=1).item()
    # 예시로 분류 결과를 문자열로 매핑
    classification_map = {0: "일반 법률 질문", 1: "복잡한 법률 질문"}
    evaluation = classification_map.get(classification, "알 수 없음")
    return evaluation

# 5. 검색 함수 (RAG)
def retrieve_relevant_info(query):
    docs = vectorstore.similarity_search(query, k=3)
    combined_docs = "\\n".join([doc.page_content for doc in docs])
    return combined_docs

# 6. LangChain의 PromptTemplate 설정
prompt_template = PromptTemplate(
    input_variables=["evaluation", "retrieved_info", "user_input"],
    template="""
    당신은 전문 법률 AI 어시스턴트입니다.

    평가 결과: {evaluation}
    검색된 정보:
    {retrieved_info}

    사용자 질문: {user_input}

    위의 평가 결과와 검색된 정보를 바탕으로 사용자에게 도움이 되는 법률 답변을 제공하세요.
    """
)

# 7. LangGraph를 사용하여 체인 구축
def create_langchain_graph():
    # koBERT 평가 노드
    def kobert_node(inputs):
        evaluation = evaluate_question_with_kobert(inputs['user_input'])
        return {'evaluation': evaluation}

    # 정보 검색 노드
    def retrieval_node(inputs):
        query = inputs['evaluation'] + " " + inputs['user_input']
        retrieved_info = retrieve_relevant_info(query)
        return {'retrieved_info': retrieved_info}

    # GPT-4o 답변 생성 노드
    def gpt_node(inputs):
        prompt = prompt_template.format(
            evaluation=inputs['evaluation'],
            retrieved_info=inputs['retrieved_info'],
            user_input=inputs['user_input']
        )
        response = gpt_llm(prompt)
        return {'response': response}

    # 체인 그래프 생성
    graph = ChainGraph()
    graph.add_node('Input', lambda inputs: {'user_input': inputs['user_input']})
    graph.add_node('koBERT Evaluation', kobert_node)
    graph.add_node('Retrieval', retrieval_node)
    graph.add_node('GPT-4o Generation', gpt_node)

    # 노드 연결
    graph.add_edge('Input', 'koBERT Evaluation')
    graph.add_edge('koBERT Evaluation', 'Retrieval')
    graph.add_edge('Retrieval', 'GPT-4o Generation')

    return graph

# 8. 파이프라인 실행 함수
def run_pipeline(user_input):
    graph = create_langchain_graph()
    inputs = {'user_input': user_input}
    outputs = graph.run(inputs)
    response = outputs['response']
    print(f"AI 응답: {response}")

# 9. 실행 예시
if __name__ == "__main__":
    user_input = input("사용자 질문: ")
    run_pipeline(user_input)
