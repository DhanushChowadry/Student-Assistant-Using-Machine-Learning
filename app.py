import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, make_response, render_template, request, redirect, url_for

courses_df = pd.read_csv("udemy_courses.csv")

courses_df["combined_features"] = courses_df["course_title"] + " " + courses_df["subject"]
courses_df["combined_features"] = courses_df["combined_features"].fillna("")
courses_df["combined_features"] = courses_df["combined_features"].str.lower()


cv = CountVectorizer()
count_matrix = cv.fit_transform(courses_df["combined_features"])


cosine_sim = cosine_similarity(count_matrix)


app = Flask(__name__)


user_ratings = {}

def recommend_courses(user_input, user_id=None):
    recommended_courses = []
    user_input = user_input.lower()
    indices = courses_df[courses_df["combined_features"].str.contains(user_input)].index
    if len(indices) == 0:
        return ["No courses found"]
    

    if user_id in user_ratings and len(user_ratings[user_id]) > 0:
        user_rated_indices = [course_id for course_id, rating in user_ratings[user_id].items()]
        sim_scores = cosine_sim[user_rated_indices]
        user_ratings_vector = np.array([rating for course_id, rating in user_ratings[user_id].items()])
        weighted_scores = sim_scores.T @ user_ratings_vector
        recommended_course_indices = np.argsort(weighted_scores)[::-1]
    else:
        index = indices[0]
        similar_courses = list(enumerate(cosine_sim[index]))
        sorted_courses = sorted(similar_courses, key=lambda x: x[1], reverse=True)[1:]
        recommended_course_indices = [course[0] for course in sorted_courses]

    for course_idx in recommended_course_indices[:10]:
        recommended_courses.append(courses_df.iloc[course_idx]["course_title"])
    return recommended_courses

 
@app.route("/", methods=["GET", "POST"])
def index():
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(len(user_ratings))
        user_ratings[user_id] = {}

    if request.method == "POST":
        user_input = request.form["user_input"]
        recommended_courses = recommend_courses(user_input, user_id)
        response = make_response(render_template("index.html", user_input=user_input, recommended_courses=recommended_courses, user_id=user_id))
        response.set_cookie('user_id', user_id)
        return response
    else:
        return render_template("index.html", user_id=user_id)

@app.route("/rate", methods=["POST"])
def rate():
    user_id = request.cookies.get('user_id')
    if not user_id:
        return redirect(url_for('index'))

    course_title = request.form["course_title"]
    rating = int(request.form["rating"])
    course_idx = courses_df[courses_df["course_title"] == course_title].index[0]
    
    if user_id in user_ratings:
        user_ratings[user_id][course_idx] = rating
    else:
        user_ratings[user_id] = {course_idx: rating}

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
