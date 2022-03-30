from autocorrect import spell
import neuspell
from neuspell import BertChecker, SclstmChecker
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SpellCorrector:

    def __init__(self, word_list):

        self.word_list = word_list
        self.new_line = " "
        self.result = []
        self.spell()
        self.result.append(self.new_line)
        print(word_list)

    def spell(self):

        for word in self.word_list:
            print("Word:", word)
            self.new_line = self.new_line + spell(str(word)) + " "
            print(self.new_line)

        return self.new_line


# class NeuSpellCorrector:

#     def __init__(self, word_list):

#         self.word_list = word_list
#         self.new_line = " "
#         self.result = []
#         self.spell()
#         self.result.append(self.new_line)

#     def spell(self):

#         checker = BertChecker()
#         checker.from_pretrained()
#         for word in self.word_list:
#             word = ''.join(word)

#         self.word_list = ' '.join(self.word_list)
#         self.new_line = checker.correct(self.word_list)
#         print(self.new_line)

#         return self.new_line


class BhuvanaSpellCorrector:

    def __init__(self, word_list):

        self.word_list = word_list
        self.new_line = " "
        self.result = []
        self.spell()
        self.result.append(self.new_line)

    def correct(self, inputs, model, tokenizer):
        input_ids = tokenizer.encode(inputs, return_tensors='pt')
        sample_output = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_p=0.99,
            num_return_sequences=1
        )
        res = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        return res

    def spell(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Bhuvana/t5-base-spellchecker")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "Bhuvana/t5-base-spellchecker")
        for word in self.word_list:
            word = ''.join(word)

        self.word_list = ' '.join(self.word_list)
        self.new_line = self.correct(self.word_list, model, tokenizer)
        print(self.new_line)

        return self.new_line
