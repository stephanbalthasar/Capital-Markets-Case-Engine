import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re

# Load the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Hidden model answer (not shown to students)
MODEL_ANSWER = """[Insert your detailed model answer here]"""

# Streamlit UI
st.set_page_config(page_title="Capital Markets Law Practice", layout="centered")
st.title("📘 Capital Markets Law Case Practice")

st.markdown("### 🧾 Case:")
st.markdown("""
Problem:
1. Neon AG is a German stock company...
2. Unicorn will transfer the licences...
3. Gerry agrees to follow Unicorn’s instructions...

---

Questions:

1. Does the conclusion of the CFA trigger capital market disclosure obligations for Neon?
2. Does this require a prospectus under the Prospectus Regulation?
3. What are the disclosure obligations for Unicorn?

Note:
You may assume all corporate authorisations are in place.
""")

# Text area for student input
student_answer = st.text_area("✍️ Your Answer", height=250)

# Button to generate feedback
if st.button("🧠 Get Feedback"):
    if student_answer.strip():
        # Encode both answers
        embeddings = model.encode([student_answer, MODEL_ANSWER], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        # Extract key legal concepts from model answer
        key_concepts = re.findall(r'\b(?:article|§{1,2})\s[\d\w\(\)]+(?:\s(?:MAR|PR|WpHG|WpÜG|MiFID II))?', MODEL_ANSWER, flags=re.IGNORECASE)
        key_concepts = list(set(key_concepts))

        # Check which concepts are present in student answer
        present_concepts = [concept for concept in key_concepts if concept.lower() in student_answer.lower()]
        missing_concepts = [concept for concept in key_concepts if concept.lower() not in student_answer.lower()]

        # Generate detailed feedback
        feedback = f"🔍 Similarity Score: {similarity:.2f}\n\n"
        feedback += "✅ Concepts you covered:\n"
        feedback += "\n".join(f"- {c}" for c in present_concepts) if present_concepts else "None detected."

        feedback += "\n\n❌ Concepts you may have missed:\n"
        feedback += "\n".join(f"- {c}" for c in missing_concepts) if missing_concepts else "None. Excellent coverage!"

        feedback += "\n\n🧠 Overall Feedback:\n"
        if similarity > 0.8:
            feedback += "Your answer demonstrates strong alignment with the model solution and covers most key legal concepts. Excellent work!"
        elif similarity > 0.5:
            feedback += "Your answer shows partial alignment with the model solution. Consider elaborating on the missing legal provisions and their implications."
        else:
            feedback += "Your answer lacks alignment with the model solution. Please review the relevant legal frameworks and ensure you address the disclosure obligations and regulatory requirements in detail."

        st.markdown("### 📋 Feedback:")
        st.info(feedback)
    else:
        st.warning("Please enter your answer before requesting feedback.")
