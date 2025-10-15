import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Hidden model answer (not shown to students)
MODEL_ANSWER = """
Under the Market Abuse Regulation, the issuer must disclose inside information as soon as possible unless a delay is justified. The issuance of convertible bonds may constitute inside information depending on its impact on the price of the securities.
"""

# Streamlit UI setup
st.set_page_config(page_title="Capital Markets Law Practice", layout="centered")
st.title("📘 Capital Markets Law Case Practice")

# Case text with numbering and line break before "Questions"
case_text = """
1. A listed company intends to issue convertible bonds in Germany.
2. The bonds will be offered to institutional investors only.
3. The company is already subject to EU disclosure obligations.

Questions:

What disclosure obligations apply under the Market Abuse Regulation?
"""

st.markdown("### 🧾 Case:")
st.markdown(case_text)

# Text area for student input
student_answer = st.text_area("✍️ Your Answer", height=250)

# Button to generate feedback
if st.button("🧠 Get Feedback"):
    if student_answer.strip():
        # Encode both answers
        embeddings = model.encode([student_answer, MODEL_ANSWER], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        # Generate detailed feedback based on similarity score
        if similarity > 0.8:
            feedback = (
                "✅ Your answer is very close to the model solution. You correctly identified the issuer's obligation to disclose inside information promptly. "
                "You also recognized that the issuance of convertible bonds may qualify as inside information depending on its market impact. Excellent work!"
            )
        elif similarity > 0.5:
            feedback = (
                "🟡 Your answer covers some key points, such as the issuer's disclosure duties. However, it could be improved by explicitly referencing the Market Abuse Regulation, "
                "and explaining how the issuance of convertible bonds might constitute inside information. Consider elaborating on the timing and conditions for disclosure."
            )
        else:
            feedback = (
                "🔴 Your answer misses several important aspects. Under the Market Abuse Regulation, issuers must disclose inside information as soon as possible unless a delay is justified. "
                "You should also consider whether the issuance of convertible bonds could affect the price of the securities, which may trigger disclosure obligations. "
                "Please review the regulation and refine your analysis."
            )

        st.markdown("### 📋 Feedback:")
        st.info(feedback)
    else:
        st.warning("Please enter your answer before requesting feedback.")
