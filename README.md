# LEX Underwriting Prediction — Streamlit Demo

## What this is
A live demo for showing your lead: upload the historical applicant dataset, the model trains on the spot,
then you score new applicants one-by-one (form) or in batch (file upload) and get:
- Decision (approved_full / approved_reduced / declined)
- Approved amount (SAR)
- Risk band
- Exception flag (when the ML model disagrees with the simple policy rule — flagged for human review)
- Identity / affordability red flags (Nafath, Yakeen, DBR stretch)

## How to run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run app.py

3. It opens in your browser (usually http://localhost:8501).

4. In the sidebar, upload the `Synthetic_Data_1_of_4.xlsx` file (or any sheet with the same columns),
   confirm the sheet name is "10K", and the model trains automatically (takes a few seconds).

5. Use the three tabs:
   - **Pattern Analysis**: segment-level approval rates, risk band validation, top decision drivers
   - **Score a New Applicant**: manual form for a single applicant, live prediction
   - **Batch Scoring**: upload a CSV/XLSX of multiple new applicants, get all of them scored + downloadable

## Important honesty notes for presenting this

- There is **no fraud label** in the dataset. The "flag" shown is an **exception/review flag**
  (model disagrees with the standard policy rule, or there's an identity/affordability red flag) —
  not a fraud-detection model. Be careful not to call it fraud detection to your lead; call it a
  review/exception flag, which is what it actually is.
- The decision/amount logic is ~98% rule-derived already (Nafath verification + risk score threshold +
  affordability cap). The ML model's real value is catching the ~1.8% exception cases where the simple
  rule and actual outcome disagree — that's the honest "lift" story.
- Forward-risk prediction (who defaults later) is NOT included in this demo because the 10K sample has too
  few default/write-off cases (22 and 126 respectively) to train a reliable model. If your lead asks about
  this, the honest answer is: needs the full 40K dataset (all 4 sheets) pooled together.
