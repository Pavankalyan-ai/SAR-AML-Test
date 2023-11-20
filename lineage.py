from utils import *

def llm_lineage(lineage_):
    if lineage_ is not None:
        li = ["Select question to get the lineage",
            "what is the customer name?",
            "what is the suspect's name?",
            "List the Merchant Name",
            "how was the bank notified?",
            "when was the bank notified?",
            "what type of fraud is taking place?",
            "when did the fraud occur?",
            "was the disputed amount greater than 5000 usd?",
            "what type of network/card is used in transaction?",
            "was the police report filed?"]
        
        
        selected_option = st.selectbox("", li)
        if selected_option in li[1:]:
            doc = lineage_[selected_option]
            for i in range(len(doc)):
                y = i+1
                st.write(f":blue[Chunk-{y}:]")
                st.write(":blue[Page Content:]", doc[i].page_content)
                st.write(":blue[Source:]",doc[i].metadata['source'])
