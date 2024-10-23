import re

from openai import OpenAI


class TextualAttractiveness:

    def __init__(self, device="mps"):
        super().__init__()
        self.device = device

        self.model = OpenAI()

    def predict(self, prompt):
        completion = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message

    def predict_bio_attractiveness(self):
        prompt = 'Rate the following online dating profile text according to the attractiveness on a scale from 0 to 10: ' \
                 'I do not like men which do not pay for me.'

        result = self.predict(prompt)
        score_texts = re.findall(' . out of 10', result.content)

        if len(score_texts) == 1:
            score_text = score_texts[0]
            return int(score_text.split()[0])
        else:
            return None

    def predict_chat_attractiveness(self):
        pass


if __name__ == "__main__":
    predictor = TextualAttractiveness()

    print(predictor.predict_bio_attractiveness())