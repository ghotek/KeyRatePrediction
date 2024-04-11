import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from data_parsing import open_fit_data

class CatModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(min_df=0.3, max_df=0.8, ngram_range=(3, 3))
        self.cat_model = CatBoostClassifier()

    @staticmethod
    def clean_text(text):
        import re
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from pymorphy3 import MorphAnalyzer

        morph = MorphAnalyzer(lang='ru')
        stop_words = set(stopwords.words('russian'))
        stop_words.update(['совет', 'директоры', 'банк', 
                           'россия', 'другой', 'это',
                           'ключевой ставка', 'директор'])
        text = re.sub('[^а-яёА-ЯЁ]', ' ', text)  # оставляем только кириллицу
        text = word_tokenize(
            text.lower(),
            language='russian')  # приводим к нижнему регистру и токенизируем по словам
        # приводим токены к нормальной форме, удаляем стоп-слова и короткие токены

        text_cleaned = []
        for token in text:
            token = token.lower().strip()
            t_clean = morph.normal_forms(token)[0]
            if len(t_clean) > 2 and (t_clean not in stop_words):
                text_cleaned.append(t_clean)
        text = " ".join(text_cleaned)  # возвращаем строку

        return text

    def fit(self, x: pd.Series, y: pd.Series) -> None:
        import pickle

        x = x.apply(self.clean_text)
        self.tfidf: TfidfVectorizer = self.tfidf.fit(x)
        x = self.tfidf.transform(x)
        self.cat_model = self.cat_model.fit(x, y)

        pickle.dump(self.tfidf, open('model_data/tfidf.pkl', 'wb'))
        pickle.dump(self.cat_model, open('model_data/model.pkl', 'wb'))

    def predict(self, x):
        import pickle

        x = x.apply(self.clean_text)
        self.tfidf = pickle.load(open('model_data/tfidf.pkl', 'rb'))
        x = self.tfidf.transform(x)
        self.cat_model = pickle.load(open('model_data/model.pkl', 'rb'))

        return self.cat_model.predict(x)
    
if __name__ == "__main__":
    data = open_fit_data()
    Cat_Model = CatModel()
    Cat_Model.fit(data['text'], data['target'])