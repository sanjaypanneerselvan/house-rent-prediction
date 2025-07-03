import streamlit as st
import pandas as pd
import joblib
import random

# Load model and feature names
model = joblib.load('model/rent_model.pkl')
features = joblib.load('model/features.pkl')

st.set_page_config(page_title="House Rent Prediction", layout="centered")

# Main title
st.title("🏠 House Rent Prediction - India 🇮🇳")
st.sidebar.header("Input Features")

# Sidebar inputs
BHK = st.sidebar.slider("BHK", 1, 5, 2)
Size = st.sidebar.slider("Size (sqft)", 300, 3000, 1000)
Bathroom = st.sidebar.slider("Bathrooms", 1, 4, 2)
City = st.sidebar.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Coimbatore", "Madurai", "Trichy", "Dindigul"])
Furnishing = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])

# Create input dictionary
input_data = {
    'BHK': BHK,
    'Size': Size,
    'Bathroom': Bathroom,
    'City_Bangalore': 0,
    'City_Chennai': 0,
    'City_Delhi': 0,
    'City_Coimbatore': 0,
    'City_Madurai': 0,
    'City_Trichy': 0,
    'City_Dindigul': 0,
    'Furnishing Status_Furnished': 0,
    'Furnishing Status_Semi-Furnished': 0,
    'Furnishing Status_Unfurnished': 0
}

input_data[f'City_{City}'] = 1
input_data[f'Furnishing Status_{Furnishing}'] = 1

df_input = pd.DataFrame([input_data])
df_input = df_input.reindex(columns=features, fill_value=0)

# Prediction
if st.button("Predict Rent"):
    result = model.predict(df_input)[0]
    st.success(f"🏡 Estimated Monthly Rent: ₹ {round(result, 2)}")

# Funny facts list
funny_facts = [
    "😆 Landlords call it 'cozy'. Tenants call it 'can’t fit a bed'.",
    "😆 Your rent increases faster than your salary!",
    "😆 Studio apartments: where your bed watches you cook.",
    "😆 Moving out? Say goodbye to your security deposit.",
    "😆 'Pet-friendly'... only if your pet is imaginary.",
    "😆 Wi-Fi included... at 2G speed.",
    "😆 You sign 20 pages to rent 200 sqft.",
    "😆 Real estate photos are pure wide-angle sorcery.",
    "😆 Your roommate’s dog has more privacy than you.",
    "😆 Visiting a rental = real-life obstacle course."
]

# Trivia list
trivia_facts = [
    "🧠 Did you know? Over 35% of urban Indians live in rented homes.",
    "🧠 Many landlords in India prefer family tenants over bachelors.",
    "🧠 Rent in metro cities can consume over 40% of monthly income.",
    "🧠 Security deposits are typically 2–10 months of rent in India.",
    "🧠 Chennai is among the cheapest metros in terms of rent.",
    "🧠 Furnished homes can cost up to 25% more than unfurnished ones.",
    "🧠 Some cities now require rent agreements to be online registered.",
    "🧠 Bangalore's Whitefield has seen a 30% rent hike post-pandemic.",
    "🧠 More Indians now prefer co-living spaces over traditional flats.",
    "🧠 Delhi NCR sees the highest fluctuation in rent pricing year-round."
]

# Enhanced random fun + trivia section
st.markdown("### 🎲 Surprise Me - Rental Fun & Facts")

if st.button("🎉 Show Me Something Interesting"):
    random_fun = random.choice(funny_facts)
    random_trivia = random.choice(trivia_facts)

    with st.container():
        st.markdown(f"""
        <div style='border: 2px solid #FFD700; border-radius: 10px; padding: 1rem; background-color: #fff8dc; margin-bottom: 10px; color: black;'>
            <b>😂 Funny Rental Fact:</b><br>{random_fun}
        </div>
        <div style='border: 2px solid #87CEFA; border-radius: 10px; padding: 1rem; background-color: #f0faff; color: black;'>
            <b>🧠 Did You Know?</b><br>{random_trivia}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style='border: 2px dashed #ccc; border-radius: 10px; padding: 1rem; margin-bottom: 10px; background-color: #f9f9f9; color: black;'>
        Click the <b>🎉 Show Me Something Interesting</b> button to get a surprise rental fun + fact combo!
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top: 2rem; margin-bottom: 1rem;">
<div style='text-align: center; padding: 10px 0; font-size: 0.9rem; color: gray;'>
    Developed by <b>Sanjay Panneerselvan</b> 💻
</div>
""", unsafe_allow_html=True)
