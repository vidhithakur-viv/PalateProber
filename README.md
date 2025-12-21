PalateProber: Amazon Fine Food Sentiment Analysis

Project Overview
PalateProber is a specialized Natural Language Processing (NLP) tool designed to investigate customer sentiment within the Amazon Fine Food marketplace. Unlike general electronics or apparel, food reviews are highly subjective and rely on sensory descriptions (taste, texture, and aroma). This project "probes" these descriptions to classify customer satisfaction and uncover the specific qualities that define premium food products.

Dataset Information
The project utilizes the Amazon Fine Food Reviews dataset, a gold-standard collection for gourmet grocery and specialty food sentiment analysis.

    Total Reviews: ~568,454

    Context: Data covers gourmet groceries, specialty snacks, and imported pantry items.

    Target Variable: Score (1-5 star ratings mapped to Positive, Neutral, and Negative sentiments).

    Input Variable: Text (Raw customer feedback including taste profiles).

Tech Stack

    Language: Python

    Environment: Jupyter Notebook / VS Code

    Libraries:

        Data Handling: Pandas, NumPy

        NLP: NLTK, Regex, BeautifulSoup

        Machine Learning: Scikit-Learn

        Visualization: Matplotlib, Seaborn

Project Workflow
1. Data Cleaning & Preprocessing

    Deduplication: Identifying and removing redundant entries from the same user on the same product.

    Text Scrubbing: Removal of HTML tags, punctuation, and special characters.

    Stopwords Removal: Eliminating words that lack emotional weight.

    Stemming: Reducing words like "delicious," "delightful," and "delicacy" to their root form ("delici").

2. Feature Extraction

    TF-IDF (Term Frequency-Inverse Document Frequency): Quantifying the importance of unique "foodie" terms across the dataset.

    Bag of Words: Transforming descriptive text into numerical vectors for the model.

3. Model Building & Evaluation

    Classification: Training supervised learning models (Naive Bayes and Logistic Regression) to distinguish between satisfied and unsatisfied customers.

    Metrics: Validating model performance using Confusion Matrices and F1-Scores.

File Structure

C:.
│   .gitignore
│   analysis.ipynb
│   analysis.py
│   README.md
│   requirements.txt
└───data
        Reviews.csv

How to Run

    Clone the repo: git clone https://github.com/YourUsername/PalateProber.git

    Install dependencies: pip install -r requirements.txt

    Run the Analysis: Open analysis.ipynb in VS Code and execute the cells.

If you find this project insightful, please consider giving it a star on GitHub.