import re
import json


def contains_non_ascii(text):
    return any(ord(char) > 127 for char in text)


def get_ascii_representation(text):
    if not text:
        return ""
    return json.dumps(text, ensure_ascii=True)[1:-1]


def clean_token(token_decode_lower):
    chars_to_strip = ' .,;:!?-_\'"\n\t\r'
    return ''.join(char for char in token_decode_lower if char not in chars_to_strip)


def num_to_word(num):
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if num == 0:
        return "zero"
    elif 1 <= num <= 9:
        return ones[num]
    elif 10 <= num <= 19:
        return teens[num - 10]
    elif 20 <= num <= 99:
        if num % 10 == 0:
            return tens[num // 10]
        else:
            return tens[num // 10] + "-" + ones[num % 10]
    elif num == 100:
        return "one hundred"
    return ""


def is_incorrect(task_name, token_decode, answer1, answer2=None, model_name=None):
    has_non_ascii_token = contains_non_ascii(token_decode) if token_decode else False
    has_non_ascii_answer1 = contains_non_ascii(answer1) if answer1 else False
    has_non_ascii_answer2 = contains_non_ascii(answer2) if answer2 else False

    token_ascii = get_ascii_representation(token_decode) if has_non_ascii_token else token_decode
    answer1_ascii = get_ascii_representation(answer1) if has_non_ascii_answer1 else answer1
    answer2_ascii = get_ascii_representation(answer2) if has_non_ascii_answer2 else answer2

    token_decode_lower = token_decode.lower() if token_decode else ""
    token_ascii_lower = token_ascii.lower() if token_ascii else ""
    answer1_lower = answer1.lower() if answer1 else ""
    answer1_ascii_lower = answer1_ascii.lower() if answer1_ascii else ""
    answer2_lower = answer2.lower() if answer2 else ""
    answer2_ascii_lower = answer2_ascii.lower() if answer2_ascii else ""
    cleaned_token_lower = clean_token(token_decode_lower)
    cleaned_token_ascii_lower = clean_token(token_ascii_lower)

    def token_exists_in_answer(token_text, token_ascii_t, answer_text, answer_ascii_t):
        return (token_text in answer_text or
                token_text in answer_ascii_t or
                token_ascii_t in answer_text or
                token_ascii_t in answer_ascii_t)

    if task_name == "task1":
        has_correct_number = any(
            s in answer1_lower or s in answer1_ascii_lower for s in ["6", "six"])
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer1_lower, answer1_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer1_lower or
                          cleaned_token_lower in answer1_ascii_lower or
                          cleaned_token_ascii_lower in answer1_lower or
                          cleaned_token_ascii_lower in answer1_ascii_lower)
        return not has_correct_number and not token_exists and cleaned_token_lower != "" and not cleaned_exists

    elif task_name == "task2":
        if has_non_ascii_token:
            char_count = len(re.findall(r'\\u[0-9a-f]{4}', token_ascii_lower))
        else:
            char_count = len(token_decode_lower)
        count_mentioned = (str(char_count) in answer1_lower or
                           str(char_count) in answer1_ascii_lower or
                           num_to_word(char_count) in answer1_lower or
                           num_to_word(char_count) in answer1_ascii_lower)
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer1_lower, answer1_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer1_lower or
                          cleaned_token_lower in answer1_ascii_lower or
                          cleaned_token_ascii_lower in answer1_lower or
                          cleaned_token_ascii_lower in answer1_ascii_lower)
        return not count_mentioned and not token_exists and cleaned_token_lower != "" and not cleaned_exists

    elif task_name in ["task3", "task4"]:
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer1_lower, answer1_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer1_lower or
                          cleaned_token_lower in answer1_ascii_lower or
                          cleaned_token_ascii_lower in answer1_lower or
                          cleaned_token_ascii_lower in answer1_ascii_lower)
        return not token_exists and not cleaned_exists and cleaned_token_lower != ""

    elif task_name == "task5":
        original_count = answer1_lower.count(token_decode_lower) + answer1_ascii_lower.count(token_decode_lower)
        ascii_count = answer1_lower.count(token_ascii_lower) + answer1_ascii_lower.count(token_ascii_lower)
        cleaned_count = answer1_lower.count(cleaned_token_lower) + answer1_ascii_lower.count(cleaned_token_lower)
        cleaned_ascii_count = (answer1_lower.count(cleaned_token_ascii_lower) +
                               answer1_ascii_lower.count(cleaned_token_ascii_lower))
        return (original_count < 2 and ascii_count < 2 and
                cleaned_count < 2 and cleaned_ascii_count < 2 and cleaned_token_lower != "")

    elif task_name == "task6":
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer1_lower, answer1_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer1_lower or
                          cleaned_token_lower in answer1_ascii_lower or
                          cleaned_token_ascii_lower in answer1_lower or
                          cleaned_token_ascii_lower in answer1_ascii_lower)
        return not token_exists and not cleaned_exists and cleaned_token_lower != ""

    elif task_name == "task7":
        if has_non_ascii_token:
            chars = re.findall(r'\\u[0-9a-f]{4}|.', token_ascii_lower)
            hyphenated = '-'.join(chars)
            chars_in_token = set(chars)
        else:
            hyphenated = '-'.join(list(token_decode_lower))
            chars_in_token = set(cleaned_token_lower)
        all_chars_present = all(
            char in answer1_lower or char in answer1_ascii_lower for char in chars_in_token)
        hyphenated_exists = hyphenated in answer1_lower or hyphenated in answer1_ascii_lower
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer1_lower, answer1_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer1_lower or
                          cleaned_token_lower in answer1_ascii_lower or
                          cleaned_token_ascii_lower in answer1_lower or
                          cleaned_token_ascii_lower in answer1_ascii_lower)
        return (not hyphenated_exists and not token_exists and
                not cleaned_exists and cleaned_token_lower != "" and not all_chars_present)

    elif task_name == "task8":
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer2_lower, answer2_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer2_lower or
                          cleaned_token_lower in answer2_ascii_lower or
                          cleaned_token_ascii_lower in answer2_lower or
                          cleaned_token_ascii_lower in answer2_ascii_lower)
        return not token_exists and not cleaned_exists and cleaned_token_lower != ""

    return False
