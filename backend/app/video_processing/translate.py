from googletrans import Translator

translator = Translator()


def translate_text(original_text, src_lang, dest_lang):
    translated = translator.translate(original_text, src=src_lang, dest=dest_lang)
    return translated.text
