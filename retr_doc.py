from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.chat_models import ChatOpenAI
from utils import *
import pickle



# Setting Env
if st.secrets["OPENAI_API_KEY"] is not None:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")   



    
# Memory setup for gpt-3.5
llm = ChatOpenAI(temperature=0.1)

def pretty_print_docs(docs):
    st.write(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def retriever2(_temp_file_path,_hf_embeddings,query):
    docs, docsearch = embedding_store(_temp_file_path,_hf_embeddings) 

    # Create a retriever from the vector store
    # retriever = docsearch.as_retriever(search_type='similarity',search_kwargs={"k": 1})
    retriever = docsearch.as_retriever(search_kwargs={"k": 2})

    _filter = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)

    compressed_docs = compression_retriever.get_relevant_documents(query)
    # st.write(compressed_docs)
    return compressed_docs

def retriever1(temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 
    results = docsearch.similarity_search("Identify name of the cardholder?", k=2, fetch_k=9)
    st.write(results)
    for doc in results:
        st.write(f"Chunk: {doc.page_content}, Source: {doc.metadata['source']}")

def retriever3(temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 
    # Create a retriever from the vector store
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    embeddings_filter = EmbeddingsFilter(embeddings=hf_embeddings, similarity_threshold=0.85)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    compressed_docs = compression_retriever.get_relevant_documents("What is the cardholder name?")
    results = "\n\n".join([d.page_content for d in compressed_docs])
    meta = [d.metadata['source'] for d in compressed_docs]
    st.write(results)
    st.write(meta)

def retriever4(temp_file_path,hf_embeddings):

    # docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 

    # Create a retriever from the vector store
    retriever = docsearch.as_retriever()

    # Create a RetrievalQA object using the LLM and the retriever
    retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)


    # Use the `apply()` or `call()` method to run the chain on a list of inputs or a single input. The inputs can be queries or prompts that you want to use to retrieve relevant documents.
    results = retrievalQA.apply(["What is the cardholder name?", "What is the fraud type?","What is the suspect name?"])

    st.write(results)


def retriever(temp_file_path,hf_embeddings):

    # if 'clicked' not in st.session_state:
    #     st.session_state.clicked = False
    
    # def set_clicked():
    #     st.session_state.clicked = True
    #     st.session_state.disabled = True

    
    lineage = st.button("Lineage", disabled=st.session_state.disabled)
    
    
    with st.spinner('Getting Information....'):
        if lineage:
            df = pd.DataFrame(
            columns=['Question','Chunk','Source'],
            index=[1,2,3,4,5,6,7,8,9,10,11])
        
            query1 = "Identify the account number and name of the cardholder?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query1)
            chunk1 =[d.page_content for d in compressed_docs]
            meta1 = [d.metadata['source'] for d in compressed_docs]  
            df.loc[1,'Question'] =  "What is the victim's name?"
            df.loc[1,'Chunk'] =  chunk1[0]
            df.loc[1,'Source'] = meta1[0]
            # st.write(chunk)
            # st.write(meta)  
        
            query2 = "What is the suspect name?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query2)
            chunk2 = "\n\n".join([d.page_content for d in compressed_docs])
            meta2 = [d.metadata['source'] for d in compressed_docs]
            df.loc[2,'Question'] = "What is the suspect's name?"   
            df.loc[2,'Chunk'] = chunk2
            df.loc[2,'Source'] = meta2[0]
            # st.write(chunk)
            # st.write(meta)  
        
            query3 = "Identify the merchant involved with the dispute?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query3)
            chunk3 = "\n\n".join([d.page_content for d in compressed_docs])
            meta3 = [d.metadata['source'] for d in compressed_docs]
            df.loc[3,'Question'] = "List the merchant name"    
            df.loc[3,'Chunk'] = chunk3
            df.loc[3,'Source'] = meta3[0]
            # st.write(chunk)
            # st.write(meta)  

            query4 = "Identify if bank was notified through email or call?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query4)
            chunk4 = "\n\n".join([d.page_content for d in compressed_docs])
            meta4 = [d.metadata['source'] for d in compressed_docs]
            df.loc[4,'Question'] = "How was the bank notified?"    
            df.loc[4,'Chunk'] = chunk4
            df.loc[4,'Source'] = meta4[0]
            # st.write(chunk)
            # st.write(meta)  

            query5 = "Identify the financial processing date/date disputed of the fraud dispute?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query5)
            chunk5 = "\n\n".join([d.page_content for d in compressed_docs])
            meta5 = [d.metadata['source'] for d in compressed_docs]
            df.loc[5,'Question'] = "When was the bank notified?"    
            df.loc[5,'Chunk'] = chunk5
            df.loc[5,'Source'] = meta5[0]
            # st.write(chunk)
            # st.write(meta)  

            query6 = "What is the Fraud Type of the dispute?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query6)
            chunk6 = "\n\n".join([d.page_content for d in compressed_docs])
            meta6 = [d.metadata['source'] for d in compressed_docs]
            df.loc[6,'Question'] = "What is the fraud type?"    
            df.loc[6,'Chunk'] = chunk6
            df.loc[6,'Source'] = meta6[0]
            # st.write(chunk)
            # st.write(meta)  

            query7 = "Identify the date of issue of the fraud?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query7)
            chunk7 = "\n\n".join([d.page_content for d in compressed_docs])
            meta7 = [d.metadata['source'] for d in compressed_docs]
            df.loc[7,'Question'] = "When did the fraud occur?"   
            df.loc[7,'Chunk'] = chunk7
            df.loc[7,'Source'] = meta7[0]
            # st.write(chunk)
            # st.write(meta)

            query8 = "Was the disputed amount greater than 5000 USD?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query8)
            chunk8 = "\n\n".join([d.page_content for d in compressed_docs])
            meta8 = [d.metadata['source'] for d in compressed_docs]
            df.loc[8,'Question'] = "Was the disputed amount greater than 5000 USD?"   
            df.loc[8,'Chunk'] = chunk8
            df.loc[8,'Source'] = meta8[0]
            # st.write(chunk)
            # st.write(meta)

            query9 = "Identify which of creadit, debit, visa network card is involved in transaction?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query9)
            chunk9 = [d.page_content for d in compressed_docs]
            meta9 = [d.metadata['source'] for d in compressed_docs]
            df.loc[9,'Question'] = "What type of cards are involved?"    
            df.loc[9,'Chunk'] = chunk9[1]
            df.loc[9,'Source'] = meta9[1]
            # st.write(chunk)
            # st.write(meta)

            query10 = "Was the police report filed?"
            compressed_docs = retriever2(temp_file_path,hf_embeddings,query10)
            chunk10 = "\n\n".join([d.page_content for d in compressed_docs])
            meta10 = [d.metadata['source'] for d in compressed_docs]
            # chunk =[]
            # for d in compressed_docs:
            #     chunk.append(d.page_content)
            
            # meta=[]
            # for d in compressed_docs:
            #     meta.append(d.metadata['source'])

            df.loc[10,'Question'] = "Was the police report filed?"    
            df.loc[10,'Chunk'] = chunk10
            df.loc[10,'Source'] = meta10[0]
            # st.write(chunk)
            # st.write(meta)

            st.table(df) 





        

