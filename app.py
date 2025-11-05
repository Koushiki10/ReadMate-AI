import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 🎨 Page Config
st.set_page_config(page_title="📚 BookMate AI - Smart Recommender", layout="centered")

# 🧠 Load Data
books = pd.read_csv("books.csv")

# 🧩 Feature Combination
books["Combined"] = books["Title"] + " " + books["Author"] + " " + books["Genre"] + " " + books["Keywords"]

# 🔢 Vectorization
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(books["Combined"])
similarity = cosine_similarity(vectors)

# 🖌️ Header
st.markdown(
    """
    <h1 style='text-align:center; color:#FF6F61; font-family:Trebuchet MS;'>📚 BookMate AI</h1>
    <h4 style='text-align:center; color:#4B8BBE;'>Your Smart Book Recommendation Assistant</h4>
    <hr style='border: 2px solid #FF6F61;'>
    """, unsafe_allow_html=True
)

# 🔍 Search Box
user_input = st.text_input("🔍 Search a Book Title:", "")

def recommend(book_title):
    if book_title not in books["Title"].values:
        return []
    index = books[books["Title"] == book_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distances[1:6]:
        recommendations.append(books.iloc[i[0]])
    return recommendations

if user_input:
    recs = recommend(user_input)
    if recs:
        st.subheader("✨ Recommended Books for You")
        for r in recs:
            st.markdown(
                f"""
                <div style='background-color:#F0F8FF; padding:10px; margin:10px 0; border-radius:10px;'>
                <b style='color:#FF6F61;'>Title:</b> {r.Title}  
                <b style='color:#4B8BBE;'>Author:</b> {r.Author}  
                <b style='color:#008080;'>Genre:</b> {r.Genre}  
                <b style='color:#555;'>Year:</b> {r.Year}
                </div>
                """, unsafe_allow_html=True
            )
    else:
        st.warning("❌ Book not found! Try another title.")
else:
    st.info("👆 Type a book name above to get recommendations!")

# 📚 Library Section
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📖 Library")
st.dataframe(books.style.set_properties(**{'background-color': '#FFFFFF', 'color': '#000000'}))
