import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

# 모델 및 토크나이저 로드
bert_tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertForSequenceClassification.from_pretrained("monologg/kobert")

gpt_tokenizer = AutoTokenizer.from_pretrained('gpt-4o-mini-2024-07-18')
gpt_model = AutoModelForCausalLM.from_pretrained('gpt-4o-mini-2024-07-18')

# GPT-4o를 LangChain의 LLM으로 래핑
from transformers import pipeline
gpt_pipeline = pipeline('text-generation', model=gpt_model, tokenizer=gpt_tokenizer)
gpt_llm = HuggingFacePipeline(pipeline=gpt_pipeline)

# 임베딩 및 벡터스토어 설정
embeddings = HuggingFaceEmbeddings("nvidia/NV-Embed-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = [
    "계약은 양 당사자 간의 합의에 의해 성립됩니다.",
    "계약의 위반 시 손해배상이 청구될 수 있습니다.",
    "계약서에 명시된 조항은 법적 구속력을 가집니다."
]
texts = []
for doc in documents:
    texts.extend(text_splitter.split_text(doc))
vectorstore = FAISS.from_texts(texts, embeddings)

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]

# 노드 함수들
def evaluate_question(state):
    messages = state['messages']
    question = messages[-1].content
    inputs = bert_tokenizer(question, return_tensors='pt')
    outputs = bert_model(**inputs)
    classification = torch.argmax(outputs.logits, dim=1).item()
    classification_map = {0: "일반 법률 질문", 1: "복잡한 법률 질문"}
    evaluation = classification_map.get(classification, "알 수 없음")
    return {"messages": [HumanMessage(content=f"평가 결과: {evaluation}")]}

def retrieve_info(state):
    messages = state['messages']
    query = messages[-1].content + " " + messages[0].content
    docs = vectorstore.similarity_search(query, k=3)
    retrieved_info = "\n".join([doc.page_content for doc in docs])
    return {"messages": [HumanMessage(content=f"검색된 정보: {retrieved_info}")]}

def generate_response(state):
    messages = state['messages']
    evaluation = messages[1].content
    retrieved_info = messages[2].content
    user_input = messages[0].content
    
    prompt_template = PromptTemplate(
        input_variables=["evaluation", "retrieved_info", "user_input"],
        template="""
        당신은 전문 법률 AI 어시스턴트입니다.

        {evaluation}
        {retrieved_info}

        사용자 질문: {user_input}

        위의 평가 결과와 검색된 정보를 바탕으로 사용자에게 도움이 되는 법률 답변을 제공하세요.
        """
    )
    
    prompt = prompt_template.format(
        evaluation=evaluation,
        retrieved_info=retrieved_info,
        user_input=user_input
    )
    response = gpt_llm(prompt)
    return {"messages": [HumanMessage(content=f"AI 응답: {response}")]}

# 그래프 생성
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("평가", evaluate_question)
    workflow.add_node("검색", retrieve_info)
    workflow.add_node("생성", generate_response)
    
    workflow.set_entry_point("평가")
    workflow.add_edge("평가", "검색")
    workflow.add_edge("검색", "생성")
    workflow.add_edge("생성", END)
    
    return workflow.compile()

# 실행 함수
def run_pipeline(user_input):
    graph = create_graph()
    inputs = {"messages": [HumanMessage(content=user_input)]}
    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "생성":
                print(value['messages'][-1].content)

# 실행 예시
if __name__ == "__main__":
    user_input = input("사용자 질문: ")
    run_pipeline(user_input)
