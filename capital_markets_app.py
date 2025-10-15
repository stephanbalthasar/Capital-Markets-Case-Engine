import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Hidden model answer (not shown to students)
MODEL_ANSWER = """
The issuer must publish a prospectus approved by the competent authority before securities are offered to the public or admitted to trading on a regulated market, unless an exemption applies.
"""

# Streamlit UI
st.set_page_config(page_title="Capital Markets Law Practice", layout="centered")
st.title("📘 Capital Markets Law Case Practice")

st.markdown("### 🧾 Case:")
st.markdown("""
A company wants to offer securities to the public in Germany. What legal steps must it take under EU prospectus regulation?
""")

# Text area for student input
student_answer = st.text_area("✍️ Your Answer", height=250)

# Button to generate feedback
if st.button("🧠 Get Feedback"):
    if student_answer.strip():
        # Encode both answers
        embeddings = model.encode([student_answer, MODEL_ANSWER], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        # Generate feedback based on similarity score
        if similarity > 0.8:
            feedback = "✅ Your answer is very close to the model solution. Well done!"
        elif similarity > 0.5:
            feedback = "🟡 Your answer covers some key points, but could be improved. Consider discussing the need for a prospectus and approval by the competent authority."
        else:
            feedback = "🔴 Your answer misses several important aspects. Review the requirements under the EU prospectus regulation."

        st.markdown("### 📋 Feedback:")
        st.info(feedback)
    else:
        st.warning("Please enter your answer before requesting feedback.")
