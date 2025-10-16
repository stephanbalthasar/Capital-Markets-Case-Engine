
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re

# Load the sentence transformer model for similarity scoring
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

similarity_model = load_similarity_model()

# Load the Hugging Face text generation pipeline for GPT-style feedback
@st.cache_resource
def load_feedback_model():
    return pipeline("text-generation", model="gpt2")

feedback_model = load_feedback_model()

# Load the model answer from the original file
def load_model_answer():
    with open("capital_markets_app (4).py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    model_answer_lines = []
    capture = False
    for line in lines:
        if 'MODEL_ANSWER = """' in line:
            capture = True
            continue
        if '"""' in line and capture:
            break
        if capture:
            model_answer_lines.append(line.strip())
    return "\n".join(model_answer_lines)

MODEL_ANSWER = load_model_answer()

# Streamlit UI
st.set_page_config(page_title="Capital Markets Law Practice", layout="centered")
st.title("📘 Capital Markets Law Case Practice")

st.markdown("### 🧾 Case:")
st.markdown("""
Problem:

Neon AG is a German stock company (Aktiengesellschaft), the shares of which have been admitted to trading on the regulated market of the Frankfurt stock exchange for a number of years. Gerry is Neon’s CEO (Vorstandsvorsitzender) and holds 25% of Neon’s shares. Gerry wants Neon to develop a new business strategy. For this, Neon would have to buy IP licences for 2.5 billion euros but has no means to afford this. Unicorn plc is a competitor of Neon’s based in the UK and owns licences of the type needed for Neon’s plans. After confidential negotiations, Unicorn, Neon, and Gerry in his personal capacity enter into a “Cooperation Framework Agreement” (“CFA”) which names all three as parties and which has the following terms:
1. Unicorn will transfer the licences to Neon by way of a capital contribution in kind (Sacheinlage). In return, Neon will increase its share capital by 30% and issue the new shares to Unicorn. The parties agree that the capital increase should take place within the next 6 months.
2. Unicorn and Gerry agree that, once the capital increase is complete, they will pre-align major decisions impacting Neon’s business strategy. Where they cannot agree on a specific measure, Gerry agrees to follow Unicorn’s instructions when voting at a shareholder meeting of Neon.
As a result of the capital increase, Gerry will hold approximately 19% in Neon, and Unicorn 23%. Unicorn, Neon and Gerry know that the agreement will come as a surprise to Neon’s shareholders, in particular, because in previous public statements, Gerry had always stressed that he wanted Neon to remain independent. They expect that the new strategy is a “game-changer” for Neon and will change its strategic orientation permanently in a substantial way.

Questions:
1. Does the conclusion of the CFA trigger capital market disclosure obligations for Neon? What is the timeframe for disclosure? Is there an option for Neon to delay disclosure?
2. Unicorn wants the new shares to be admitted to trading on the regulated market in Frankfurt. Does this require a prospectus under the Prospectus Regulation? What type of information in connection with the CFA would have to be included in such a prospectus?
3. What are the capital market law disclosure obligations that arise for Unicorn once the capital increase and admission to trading are complete and Unicorn acquires the new shares? Can Unicorn participate in Neon’s shareholder meetings if it does not comply with these obligations?

Note:
Your answer will not have to consider the SRD, §§ 111a–111c AktG, or EU capital market law that is not included in your permitted material. You may assume that Gerry and Neon have all corporate authorisations for the conclusion of the CFA and the capital increase.
""")

# Text area for student input
student_answer = st.text_area("✍️ Your Answer", height=250)

# Button to generate feedback
if st.button("🧠 Get Feedback"):
    if student_answer.strip():
        # Compute similarity score
        embeddings = similarity_model.encode([student_answer, MODEL_ANSWER], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        # Extract key legal concepts from model answer
        key_concepts = re.findall(r'\b(?:article|§{1,2})\s[\d\w\(\)]+(?:\s(?:MAR|PR|WpHG|WpÜG|MiFID II))?', MODEL_ANSWER, flags=re.IGNORECASE)
        key_concepts = list(set(key_concepts))

        # Check which concepts are present in student answer
        present_concepts = [concept for concept in key_concepts if concept.lower() in student_answer.lower()]
        missing_concepts = [concept for concept in key_concepts if concept.lower() not in student_answer.lower()]

        # Generate GPT-style feedback using Hugging Face model
        prompt = f"Compare the following student answer to the model answer. Identify missing legal concepts, incorrect reasoning, and explain the relevance of each. Then summarize the overall quality of the answer.\n\nStudent Answer:\n{student_answer}\n\nModel Answer:\n{MODEL_ANSWER}\n\nFeedback:"
        gpt_feedback = feedback_model(prompt, max_length=512, do_sample=True)[0]['generated_text']

        # Display feedback
        st.markdown("### 📊 Similarity Score:")
        st.info(f"{similarity:.2f}")

        st.markdown("### ✅ Concepts You Covered:")
        st.success("\n".join(f"- {c}" for c in present_concepts) if present_concepts else "None detected.")

        st.markdown("### ❌ Concepts You May Have Missed:")
        st.error("\n".join(f"- {c}" for c in missing_concepts) if missing_concepts else "None. Excellent coverage!")

        st.markdown("### 🧠 GPT-Based Feedback:")
        st.info(gpt_feedback)

        # Chatbot interface
        st.markdown("### 💬 Ask a Follow-Up Question:")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.chat_input("Ask a question about the feedback...")
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            chat_prompt = f"Student asked: {user_question}\nBased on the previous feedback and model answer, provide a helpful explanation."
            chat_response = feedback_model(chat_prompt, max_length=256, do_sample=True)[0]['generated_text']
            st.session_state.chat_history.append(("bot", chat_response))

        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").markdown(msg)
            else:
                st.chat_message("assistant").markdown(msg)
    else:
        st.warning("Please enter your answer before requesting feedback.")
