from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)


# ==========================================
# TRAINED AI MODEL LOAD
# ==========================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# DOG BEHAVIOUR ANALYSE
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()


    # ==========================================
    # USER KE SELECTED SIGNS
    # ==========================================

    features = [
        int(data.get("limping", 0)),
        int(data.get("low_activity", 0)),
        int(data.get("paw_licking", 0)),
        int(data.get("scratching", 0)),
        int(data.get("eating_less", 0)),
        int(data.get("unusual_sounds", 0))
    ]


    # ==========================================
    # AI PREDICTION
    # ==========================================

    prediction = model.predict([features])[0]


    # ==========================================
    # AI PREDICTION CONFIDENCE
    # ==========================================

    probabilities = model.predict_proba([features])[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )


    # ==========================================
    # USER NE KAUNSE SIGNS SELECT KIYE
    # ==========================================

    signs = []


    if features[0] == 1:
        signs.append("chalne mein dikkat")


    if features[1] == 1:
        signs.append("normal se kam activity")


    if features[2] == 1:
        signs.append("paw ko baar-baar lick karna")


    if features[3] == 1:
        signs.append("baar-baar scratching karna")


    if features[4] == 1:
        signs.append("khana kam khana")


    if features[5] == 1:
        signs.append("unusual sound / zyada awaz karna")


    # ==========================================
    # RESULT INFORMATION
    # ==========================================

    result_info = {

        "Normal": {

            "title": "Koi unusual sign nahi mila",

            "message": (
                "Aapne jo behaviour select kiya hai, "
                "uske basis par dog ka behaviour mostly normal lag raha hai."
            ),

            "possible_reason": (
                "Abhi kisi major discomfort ka clear sign nahi mila."
            ),

            "action": [
                "Dog ki normal routine continue rakhein.",
                "Fresh water aur proper food available rakhein.",
                "Agar behaviour suddenly change ho to dobara observe karein."
            ],

            "avoid": [
                "Bina reason ke medicine na dein."
            ],

            "when_vet": (
                "Agar dog ka behaviour suddenly change ho ya problem "
                "continue rahe, to vet se baat karein."
            )
        },


        "Mild Concern": {

            "title": "Thodi dikkat ke signs mile",

            "message": (
                "Dog ke behaviour mein kuch unusual changes dikh rahe hain."
            ),

            "possible_reason": (
                "Ye tiredness, halka discomfort, irritation ya temporary "
                "problem ki wajah se ho sakta hai."
            ),

            "action": [
                "Dog ko thoda rest dein.",
                "Uske behaviour ko observe karein.",
                "Fresh water available rakhein.",
                "Dekhein ki problem kuch time mein kam hoti hai ya nahi."
            ],

            "avoid": [
                "Dog ko forcefully run ya exercise na karayein.",
                "Human medicine khud se na dein."
            ],

            "when_vet": (
                "Agar signs continue rahein ya badh jaayein, "
                "to veterinarian se contact karein."
            )
        },


        "Moderate Concern": {

            "title": "Kuch important signs mile",

            "message": (
                "Dog mein ek se zyada unusual behaviour signs dikh rahe hain."
            ),

            "possible_reason": (
                "Ye physical discomfort, chot, irritation, pain ya "
                "kisi aur health problem ki taraf ishara kar sakta hai."
            ),

            "action": [
                "Dog ko proper rest dein.",
                "Agar paw ya body par problem dikh rahi hai to carefully observe karein.",
                "Running aur jumping kuch time ke liye avoid karein.",
                "Food aur water intake par dhyan dein."
            ],

            "avoid": [
                "Human painkiller ya koi medicine khud se na dein.",
                "Apne najdiki Animal doctor ko dikhayein.",
                "Dog ko forcefully walk ya exercise na karayein.",
                "Problem ko ignore na karein agar ye continue ho."
            ],

            "when_vet": (
                "Agar problem continue ho, badh rahi ho, "
                "ya dog ko zyada discomfort ho raha ho, "
                "to veterinarian se contact karna better hai."
            )
        },


        "High Concern": {

            "title": "Kai unusual signs mile – dhyan dena zaroori hai",

            "message": (
                "Dog mein kai unusual behaviour signs ek saath dikh rahe hain."
            ),

            "possible_reason": (
                "Ye significant discomfort, injury, pain ya kisi health "
                "problem ki possibility dikha sakta hai."
            ),

            "action": [
                "Dog ko calm aur comfortable rakhein.",
                "Usko unnecessary movement se bachayein.",
                "Food, water aur behaviour ko closely observe karein.",
                "Jaldi veterinarian se advice lene par consider karein."
            ],

            "avoid": [
                "Human medicines bilkul khud se na dein.",
                "Apne najdiki Animal doctor ko dikhayein.",
                "Dog ko forcefully walk ya exercise na karayein.",
                "Online result ko final diagnosis na samjhein."
            ],

            "when_vet": (
                "Agar dog bahut uncomfortable hai, chal nahi paa raha, "
                "khana-peena bahut kam kar raha hai, ya problem rapidly "
                "badhti hai, to veterinarian se jaldi contact karein."
            )
        }
    }


    info = result_info[prediction]


    # ==========================================
    # AI EXPLANATION
    # ==========================================

    # Kitne unusual signs select kiye gaye
    sign_count = len(signs)


    # AI explanation
    if sign_count == 0:

        ai_explanation = (
            "AI model ko koi unusual behaviour sign input ke roop mein nahi mila."
        )


    elif sign_count == 1:

        ai_explanation = (
            "AI model ko 1 unusual behaviour sign input ke roop mein mila. "
            "Model ne is input ke basis par result predict kiya."
        )


    elif sign_count <= 3:

        ai_explanation = (
            f"AI model ko {sign_count} unusual behaviour signs "
            "input ke roop mein mile. "
            "Model ne selected behaviour inputs ke basis par result predict kiya."
        )


    else:

        ai_explanation = (
            f"AI model ko {sign_count} unusual behaviour signs "
            "input ke roop mein mile. "
            "Multiple behaviour inputs ke basis par model ne result predict kiya."
        )


    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    feature_names = [
        "Limping",
        "Low Activity",
        "Paw Licking",
        "Scratching",
        "Eating Less",
        "Unusual Sounds"
    ]


    feature_importance = []


    for name, importance in zip(
        feature_names,
        model.feature_importances_
    ):

        feature_importance.append({

            "name": name,

            "importance": round(
                float(importance) * 100,
                2
            )

        })


    # Highest importance first
    feature_importance.sort(
        key=lambda x: x["importance"],
        reverse=True
    )


    # ==========================================
    # FINAL RESPONSE WEBSITE KO
    # ==========================================

    return jsonify({

        "prediction": prediction,

        "confidence": confidence,

        "title": info["title"],

        "message": info["message"],

        "possible_reason": info["possible_reason"],

        "signs": signs,

        "sign_count": sign_count,

        "ai_explanation": ai_explanation,

        "feature_importance": feature_importance,

        "action": info["action"],

        "avoid": info["avoid"],

        "when_vet": info["when_vet"]

    })


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)