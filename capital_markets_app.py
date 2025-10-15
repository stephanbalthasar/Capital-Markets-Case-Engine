import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Hidden model answer (not shown to students)
MODEL_ANSWER = """
Model Answer:
1. Question 1 requires a discussion of whether the conclusion of the CFA triggers an obligation to publish “inside information” pursuant to article 17(1) MAR. 
a) On the facts of the case (i.e., new shareholder structure of Neon, combined influence of Gerry and Unicorn, substantial change of strategy, etc.), students would have to conclude that the conclusion of the CFA is inside information within the meaning of article 7(1)(a): 
aa) It relates to an issuer (Neon) and has not yet been made public.
bb) Even if the agreement depends on further implementing steps, it creates information of a precise nature within the meaning of article 7(2) MAR in that there is an event that has already occurred – the conclusion of the CFA –, which is sufficient even if one considered it only as an “intermediate step” of a “protracted process”. In addition, subsequent events – the capital increase – can “reasonably be expected to occur” and therefore also qualify as information of a precise nature. A good answer would discuss the “specificity” requirement under article 7(2) MAR and mention that pursuant to the ECJ decision in Lafonta, it is sufficient for the information to be sufficiently specific to constitute a basis on which to assess the effect on the price of the financial instruments, and that the only information excluded by the specificity requirement is information that is “vague or general”. Also, the information is something a reasonable investor would likely use, and therefore likely to have a significant effect on prices within the meaning of article 7(4) MAR.
cc) The information “directly concerns” the issuer in question. As a result, article 17(1) MAR requires Neon to “inform the public as soon as possible”. Students should mention that this allows issuers some time for fact-finding, but otherwise, immediate disclosure is required. Delay is only possible under article 17(4) MAR. However, there is nothing to suggest that Neon has a legitimate interest within the meaning of article 17(4)(a), and at any rate, given previous communication by Neon, a delay would be likely to mislead the public within the meaning of article 17(4)(b). Accordingly, a delay could not be justified under article 17(4) MAR.
Notabene: Subscribing to new shares not yet issued (as done in the CFA) does not trigger any disclosure obligations under §§ 38(1), 33(3) WpHG. At any rate, these would only be incumbent on Unicorn, not Neon. 
2. Question 2 requires an analysis of prospectus requirements under the Prospectus Regulation.
a) There is no public offer within the meaning of article 2(d) PR that would trigger a prospectus requirement under article 3(1) PR. However, pursuant to article 3(3) PR, admission of securities to trading on a regulated market requires prior publication of a prospectus. Neon shares qualify as securities under article 2(a) PR in conjunction with article 4(1)(44) MiFID II. Students should discuss the fact that there is an exemption for this type of transaction under article 1(5)(a) PR, but that the exemption is limited to a capital increase of 20% or less so does not cover Neon’s case. Accordingly, admission to trading requires publication of a prospectus (under article 21 PR), which in turn makes it necessary to have the prospectus approved under article 20(1) PR). A very complete answer would mention that Neon could benefit from the simplified disclosure regime for secondary issuances under article 14(1)(a) PR.
b) As regards the content of the prospectus, students are expected to explain that the prospectus would have to include all information in connection with the CFA that is material within the meaning of article 6(1) PR, in particular, as regards the prospects of Neon (article 6(1)(1)(a) PR) and the reasons for the issuance (article 6(1)(1)(c) PR). The prospectus would also have to describe material risks resulting from the CFA and the new strategy (article 16(1) PR). A good answer would mention that the “criterion” for materiality under German case law is whether an investor would “rather than not” use the information for the investment decision.
3. The question requires candidates to address disclosure obligations under the Transparency Directive and the Takeover Bid Directive and implementing domestic German law. 
a) As Neon’s shares are listed on a regulated market, Neon is an issuer within the meaning of § 33(4) WpHG, so participations in Neon are subject to disclosure under §§33ff. WpHG. Pursuant to § 33(1) WpHG, Unicorn will have to disclose the acquisition of its stake in Neon. The relevant position to be disclosed includes the 23% stake held by Unicorn directly. In addition, Unicorn will have to take into account Gerry’s 19% stake if the CFA qualifies as “acting in concert” within the meaning of § 34(2) WpHG. In this context, students should differentiate between the two types of acting in concert, namely (i) an agreement to align the exercise of voting rights which qualifies as acting in concert irrespectively of the impact on the issuer’s strategy, and (ii) all other types of alignment which only qualify as acting in concert if it is aimed at modifying substantially the issuer’s strategic orientation. On the facts of the case, both requirements are fulfilled. A good answer should discuss this in the light of the BGH case law, and ideally also consider whether case law on acting in concert under WpÜG can and should be used to assess acting in concert under WpHG. A very complete answer would mention that Unicorn also has to make a statement of intent pursuant to § 43(1) WpHG.
b) The acquisition of the new shares is also subject to WpÜG requirements pursuant to § 1(1) WpÜG as the shares issued by Neon are securities within the meaning of § 2(2) WpÜG and admitted to trading on a regulated market. Pursuant to § 35(1)(1) WpÜG, Unicorn has to disclose the fact that it acquired “control” in Neon and publish an offer document submit a draft offer to BaFin, §§ 35(2)(1), 14(2)(1) WpÜG. “Control” is defined as the acquisition of 30% or more in an issuer, § 29(2) WpÜG. The 23% stake held by Unicorn directly would not qualify as “control” triggering a mandatory bid requirement. However, § 30(2) WpÜG requires to include in the calculation shares held by other parties with which Unicorn is acting in concert, i.e., Gerry’s 19% stake (students may refer to the discussion of acting in concert under § 34(2) WpHG). The relevant position totals 42% and therefore the disclosure requirements under § 35(1) WpÜG.
c) Failure to disclose under § 33 WpHG/§ 35 WpÜG will suspend Unicorn’s shareholder rights under § 44 WpHG, § 59 WpÜG. No such sanction exists as regards failure to make a statement of intent under § 43(1) WpHG.

"""

# Streamlit UI setup
st.set_page_config(page_title="Capital Markets Law Practice", layout="centered")
st.title("📘 Capital Markets Law Case Practice")

# Case text with numbering and line break before "Questions"
case_text = """
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
