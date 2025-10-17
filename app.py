import streamlit as st
import os
import base64
import fitz  # PyMuPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Constants for asset paths
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
CASE_FILES = {
    "Case 1": os.path.join(ASSETS_DIR, "case1.txt"),
    "Case 2": os.path.join(ASSETS_DIR, "case2.txt")
}
MODEL_ANSWERS = {
    "Case 1": os.path.join(ASSETS_DIR, "model_answer1.txt"),
    "Case 2": os.path.join(ASSETS_DIR, "model_answer2.txt")
}
COURSE_MANUAL_PATH = os.path.join(ASSETS_DIR, "EUCapML - Course Booklet.pdf")

# Load course manual text from PDF
def load_course_manual_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load text from a file
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Generate feedback using Groq API (placeholder function)
def generate_feedback(student_answer, model_answer, course_manual_text, case_text):
    # This is a placeholder. You would integrate Groq API here using st.secrets["GROQ_API_KEY"]
    feedback = f"""
    ✅ Covered Well: [Placeholder analysis]
    ❌ Incorrect: [Placeholder analysis]
    ❓ Missing: [Placeholder analysis]
    📘 Suggestions: [Placeholder suggestions]
    """
    return feedback

# Send email with answer and feedback
def send_email(recipient_email, student_answer, feedback):
    sender_email = "your_email@example.com"
    subject = "EUCapML Case Tutor - Your Submission and Feedback"
    body = f"Student Answer:\n\n{student_answer}\n\nFeedback:\n\n{feedback}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("localhost") as server:
            server.send_message(msg)
        return True
    except Exception as e:
        return False

# Chatbot response (placeholder)
def chatbot_response(user_input, case_text, feedback, course_manual_text):
    return f"Chatbot response to: '{user_input}' [Placeholder]"

# Streamlit App
def main():
    st.set_page_config(page_title="EUCapML Case Tutor", layout="wide")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("EUCapML Case Tutor Login")
        pin_input = st.text_input("Enter your STUDENT_PIN", type="password")
        if st.button("Login"):
            if pin_input == st.secrets["STUDENT_PIN"]:
                st.session_state.authenticated = True
                st.experimental_set_query_params(dummy=str(pin_input))  # Triggers rerun
            else:
                st.error("Invalid PIN. Please try again.")
        return

    # Main App Interface
    st.image(LOGO_PATH, width=150)
    st.title("EUCapML Case Tutor")

    case_choice = st.selectbox("Select a case", list(CASE_FILES.keys()))
    case_text = load_text(CASE_FILES[case_choice])
    st.subheader(f"{case_choice} Details")
    st.text_area("Case Content", case_text, height=300, disabled=True)

    student_answer = st.text_area("Enter your answer here", height=200)
    if st.button("Generate Feedback"):
        model_answer = load_text(MODEL_ANSWERS[case_choice])
        course_manual_text = load_course_manual_text(COURSE_MANUAL_PATH)
        feedback = generate_feedback(student_answer, model_answer, course_manual_text, case_text)
        st.session_state.feedback = feedback
        st.subheader("Feedback")
        st.text_area("Feedback", feedback, height=300, disabled=True)

    if "feedback" in st.session_state:
        email_input = st.text_input("Enter your email to receive feedback")
        if st.button("Send Email"):
            success = send_email(email_input, student_answer, st.session_state.feedback)
            if success:
                st.success("Email sent successfully!")
            else:
                st.error("Failed to send email.")

    st.subheader("Chatbot Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask a question")
    if st.button("Send"):
        course_manual_text = load_course_manual_text(COURSE_MANUAL_PATH)
        feedback = st.session_state.get("feedback", "")
        response = chatbot_response(user_query, case_text, feedback, course_manual_text)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("TutorBot", response))

    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
