import re

import config

from openai import OpenAI


class TextualAttractiveness:

    def __init__(self):
        self.model = OpenAI(api_key=config.open_ai_key)

    def predict(self, prompt):
        completion = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message

    def predict_bio_attractiveness(self, bio: str, gender):
        prompt = (f'Rate the following online dating profile text according to the attractiveness on a scale from 0 to 10: {bio}. '
                  f'Only output the score.')

        result = self.predict(prompt)

        try:
            return int(result.content)
        except ValueError:
            print(f'Error: {result.content}')

    def predict_chat_attractiveness(self):
        pass


if __name__ == "__main__":
    predictor = TextualAttractiveness()

    print(predictor.predict_bio_attractiveness())
