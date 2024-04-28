from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
import os
import shutil
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

data = """Introduction:
Adverse events (AEs) are unwanted or harmful effects that occur in patients during clinical trials. AEs can range from mild to severe and can impact patient safety, study outcomes, and regulatory approval. This report describes a case of a serious adverse event observed in a phase I clinical trial of a new medication for the treatment of advanced solid tumors.

Case Description:
The patient, a 42-year-old female with metastatic breast cancer, was enrolled in the phase I clinical trial and received the study medication at a dose of 500 mg once daily. The patient had a significant medical history, including a previous diagnosis of hepatitis B and C, but her liver function tests were normal at the time of enrollment. The patient was also taking several other medications, including tamoxifen and levothyroxine.

On day 7 of treatment, the patient reported severe abdominal pain, nausea, and vomiting. The patient's vital signs were stable, but a physical examination revealed diffuse abdominal tenderness. Laboratory testing revealed elevated liver enzymes, total bilirubin, and international normalized ratio (INR). A CT scan of the abdomen showed evidence of hepatic necrosis.

Investigations:
The patient was admitted to the hospital for further evaluation and management. The study medication was discontinued, and the patient was started on supportive care, including intravenous fluids, pain management, and nutritional support. Further investigations, including a liver biopsy, confirmed the diagnosis of drug-induced liver injury.

Discussion:
Drug-induced liver injury is a known adverse effect of many medications, including some chemotherapeutic agents. The incidence of drug-induced liver injury is higher in patients with pre-existing liver disease or those taking other medications that affect the liver. In this case, the patient developed severe drug-induced liver injury shortly after starting the study medication. The patient had a history of hepatitis B and C, which may have predisposed her to liver injury. Additionally, the patient was taking several other medications, which may have contributed to the development of liver injury.

Conclusion:
This case illustrates a serious adverse event that occurred during a phase I clinical trial of a new medication for the treatment of advanced solid tumors. The patient developed severe drug-induced liver injury, likely related to the study medication. Clinicians and investigators involved in clinical trials must be vigilant for potential adverse events, especially in the early phases of drug development. Prompt recognition and management of adverse events are essential to prevent further harm and improve patient outcomes. In this case, the study medication was discontinued, and the patient received supportive care. The patient's liver function gradually improved over several weeks, and she was able to resume treatment for her metastatic breast cancer.

However, this case also raises important questions about the inclusion criteria for phase I clinical trials and the potential risks associated with enrolling patients with pre-existing medical conditions or those taking other medications. The patient in this case had a history of hepatitis B and C, which may have increased her risk of developing drug-induced liver injury. The patient was also taking several other medications, which may have contributed to the development of liver injury.

In order to mitigate the risk of adverse events in phase I clinical trials, it may be necessary to develop more rigorous inclusion and exclusion criteria that take into account the potential risks associated with pre-existing medical conditions or concomitant medications. Additionally, close monitoring of patients during clinical trials, including regular laboratory testing and imaging studies, may be necessary to ensure the early detection and management of adverse events.

In conclusion, adverse events are an inherent risk of clinical trials, especially in the early phases of drug development. Clinicians and investigators must remain vigilant for potential adverse events and take steps to mitigate the risks associated with enrolling patients with pre-existing medical conditions or those taking other medications. Prompt recognition and management of adverse events are essential to prevent further harm and improve patient outcomes. The importance of patient safety cannot be overemphasized, and all efforts must be made to minimize the risks associated with clinical trials.

In addition to rigorous inclusion and exclusion criteria and close monitoring of patients during clinical trials, it may also be necessary to consider alternative trial designs, such as adaptive designs or Bayesian approaches, which allow for greater flexibility in the trial design and can potentially reduce the risk of adverse events. These trial designs can be particularly useful in the early phases of drug development when there is limited information on the safety and efficacy of the drug.

Furthermore, the reporting and analysis of adverse events in clinical trials are essential to improving patient safety and informing future drug development. All adverse events should be promptly reported to the study sponsor and regulatory authorities, and a thorough analysis of the data should be conducted to identify any potential trends or patterns. This analysis can help to identify potential risk factors for adverse events and inform future trial designs or inclusion and exclusion criteria.

In conclusion, adverse events are an inherent risk of clinical trials, and clinicians and investigators must remain vigilant for potential adverse events and take steps to mitigate the risks associated with enrolling patients with pre-existing medical conditions or those taking other medications. The reporting and analysis of adverse events are essential to improving patient safety and informing future drug development. Despite the challenges associated with clinical trials, they remain an essential component of drug development and are necessary to bring new treatments to patients in need."""

"""
What is the name and dose of the drug(s) involved in the ADR?
What is the patient’s age, gender, and medical history?
What are the symptoms experienced by the patient as a result of the ADR?
What was the outcome of the ADR, including any medical intervention required?
Was the ADR expected or unexpected based on the known side effects of the drug?
Was the ADR related to the drug or caused by another factor?
What action was taken to manage or prevent the ADR from occurring again?
What is the severity of the ADR, and is it considered serious or life-threatening?
Has the ADR been previously reported, and if so, what was the outcome of previous reports?
Are there any other factors that may have contributed to the ADR, such as the patient’s lifestyle or other medications being taken?"""

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


class Chatbot:
    def __init__(self):
        self.retriever = self.get_retriver()

    def get_retriver(self):
        """
        Get the retriever for the chatbot
        """
        try:
            embedding = OpenAIEmbeddings()
            
            vectordb = FAISS.load_local(
                # "HospitalManagement\vectorstore\db_faiss",
                "vectorstore/db_faiss",
                embedding,
                allow_dangerous_deserialization=True,
            )

            retriever = vectordb.as_retriever(k=2)
            print("Retriever set")
            return retriever
        except Exception as e:
            print(e)
            raise e

    def ask_que(self, query):
        """
        Ask a question to the chatbot
        """

        # try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            verbose = True,
        )
        response = qa_chain( query)
        return response
        # except Exception as e:
        #     print(e)
        #     raise e


# obj = Chatbot()
# print(obj.ask_que("Was the ADR expected or unexpected based on the known side effects of the drug?"))
