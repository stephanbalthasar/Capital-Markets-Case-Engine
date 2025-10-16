import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
    return embedder, generator

embedder, generator = load_models()

# Embedded model answer
MODEL_ANSWER = """PASTE YOUR MODEL ANSWER HERE"""

# Streamlit UI
st.set_page_config(page_title="Capital Markets Law Practice", layout="centered")
st.title("📘 Capital Markets Law Case Practice")

st.markdown("### 🧠 Case:")
st.markdown("""
Neon AG is a German stock company (Aktiengesellschaft), the shares of which have been admitted to trading on the regulated market of the Frankfurt stock exchange for a number of years. Gerry is Neon’s CEO (Vorstandsvorsitzender) and holds 25% of Neon’s shares. Gerry wants Neon to develop a new business strategy. For this, Neon would have to buy IP licences for 2.5 billion euros but has no means to afford this. Unicorn plc is a competitor of Neon’s based in the UK and owns licences of the type needed for Neon’s plans. After confidential negotiations, Unicorn, Neon, and Gerry in his personal capacity enter into a “Cooperation Framework Agreement” (“CFA”) which names all three as parties and which has the following terms:
1. Unicorn will transfer the licences to Neon by way of a capital contribution in kind (Sacheinlage). In return, Neon will increase its share capital by 30% and issue the new shares to Unicorn. The parties agree that the capital increase should take place within the next 6 months.
2. Unicorn and Gerry agree that, once the capital increase is complete, they will pre-align major decisions impacting Neon’s business strategy. Where they cannot agree on a specific measure, Gerry agrees to follow Unicorn’s instructions when voting at a shareholder meeting of Neon.
As a result of the capital increase, Gerry will hold approximately 19% in Neon, and Unicorn 23%. Unicorn, Neon and Gerry know that the agreement will come as a surprise to Neon’s shareholders, in particular, because in previous public statements, Gerry had always stressed that he wanted Neon to remain independent. They expect that the new strategy is a “game-changer” for Neon and will change its strategic orientation permanently in a substantial way.

Questions:
1. Does the conclusion of the CFA trigger capital market disclosure obligations for Neon? What is the timeframe for disclosure? Is there an option for Neon to delay disclosure?
2. Unicorn wants the new shares to be admitted to trading on the regulated market in Frankfurt. Does this require a prospectus under the Prospectus Regulation? What type of information in connection with the CFA would have to be included in such a prospectus?
3. What are the capital market law disclosure obligations that arise for Unicorn once the capital increase and admission to trading are complete and Unicorn acquires the new shares? Can Unicorn participate in Neon’s shareholder meetings if it does not comply with these obligations?
""")

# Student input
student_answer = st.text_area("✍️ Your Answer", height=250)

# Feedback generation
if st.button("🧠 Get Feedback"):
    if student_answer.strip():
        # Similarity scoring
        embeddings = embedder.encode([student_answer, MODEL_ANSWER], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        # Concept extraction
        key_concepts = re.findall(r'\b(?:article|§{1,2})\s[\d\w\(\)]+(?:\s(?:MAR|PR|WpHG|WpÜG|MiFID II))?', MODEL_ANSWER, flags=re.IGNORECASE)
        key_concepts = list(set(key_concepts))
        present_concepts = [c for c in key_concepts if c.lower() in student_answer.lower()]
        missing_concepts = [c for c in key_concepts if c.lower() not in student_answer.lower()]

        # GPT-style feedback generation
        prompt = f"Compare the following student answer to the model answer. Identify incorrect reasoning, explain why missing legal concepts are important, and summarize the overall quality.\n\nStudent Answer:\n{student_answer}\n\nModel Answer:\n{MODEL_ANSWER}\n\nFeedback:"
        gpt_feedback = generator(prompt, max_length=512, do_sample=True)[0]['generated_text']

        # Display feedback
        st.markdown("### 📋 Feedback:")
        st.info(f"🔍 Similarity Score: {similarity:.2f}\n\n✅ Concepts Covered:\n" + ("\n".join(f"- {c}" for c in present_concepts) if present_concepts else "None detected.") + "\n\n❌ Concepts Missed:\n" + ("\n".join(f"- {c}" for c in missing_concepts) if missing_concepts else "None. Excellent coverage!") + "\n\n🧠 GPT Feedback:\n" + gpt_feedback)
    else:
        st.warning("Please enter your answer before requesting feedback.")

# Chatbot interface
st.markdown("### 💬 Ask Follow-Up Questions")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message("user").markdown(msg["user"])
    st.chat_message("assistant").markdown(msg["bot"])

user_input = st.chat_input("Ask a question about the case or feedback...")
if user_input:
    st.session_state.chat_history.append({"user": user_input, "bot": "Thinking..."})
    prompt = f"You are a legal tutor. Answer the student's follow-up question based on the case and feedback.\n\nQuestion: {user_input}\n\nCase:\n{MODEL_ANSWER}\n\nAnswer:"
    response = generator(prompt, max_length=256, do_sample=True)[0]['generated_text']
    st.session_state.chat_history[-1]["bot"] = response
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(response)
