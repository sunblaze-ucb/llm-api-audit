import random
import argparse
import json
import os
import re
from tqdm import tqdm
from openai import OpenAI

def _remove_special_characters(text):
    # remove enumeration
    text = re.sub(r'\d+\.', '', text)

    # remove all special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # replace indentation with a single space
    text = re.sub(r'\s+', ' ', text)

    return text

def remove_text(text):
    lines = text.split('\n')

    output_lines = []
    in_code_block = False

    # detect triple-backtick code blocks (including optional language).
    code_block_regex = re.compile(r'^\s*```')

    # detect headings (e.g. "#", "##", etc.) or bullet/numbered list items.
    heading_or_list_regex = re.compile(r'^(\s{0,3}(?:[#]+|(?:\d+\.)|[-*+]))(\s*)(.*)')

    for line in lines:
        # check for entering/exiting a triple-backtick code block
        if code_block_regex.search(line):
            # toggle code block state
            if not in_code_block:
                # enter a code block
                output_lines.append(line)  # Keep the backtick line as is
                in_code_block = True
            else:
                # exit a code block
                output_lines.append(line)  # Keep the backtick line as is
                in_code_block = False

            continue

        if in_code_block:
            # inside code blocks, replace all non-empty lines with 'xxx' but preserve indentation.
            if line.strip() == "":
                # blank line in code block, keep it
                output_lines.append(line)
            else:
                # non-empty line => replace with 'xxx' (preserve leading indentation)
                leading_spaces = len(line) - len(line.lstrip())
                output_lines.append(' ' * leading_spaces + 'xxx')
            continue
        else:
            # normal line (outside code blocks).

            # 1) detect heading or list marker
            match = heading_or_list_regex.match(line)
            if match:
                marker_part = match.group(1)  # e.g. "#", "1.", "-", "*"
                spacing_part = match.group(2) # e.g. " " or multiple spaces
                rest_part = match.group(3)    # the rest of the line

                # replace the rest_part with a version that has inline markdown replaced by 'xxx'
                replaced_rest = replace_inlines_with_xxx(rest_part)

                # then replace any plain text left outside inline markers with 'xxx'
                output_lines.append(f"{marker_part}{spacing_part}{replaced_rest}")
            else:
                # no heading/list marker, so just replace inlines with 'xxx'
                replaced_line = replace_inlines_with_xxx(line)
                output_lines.append(replaced_line)

    # re-join the transformed lines
    output_lines = [line for line in output_lines if line != ""]
    # merge consecutive xxx into a single xxx
    tmp = []
    i = 0
    while i < len(output_lines):
        j = i
        while j < len(output_lines) and (output_lines[j] == "xxx" or output_lines[j] == ""):
            j += 1
        if j - i > 1:
            tmp.append("xxx")
            i = j
        else:
            tmp.append(output_lines[i])
            i += 1
    return "\n".join(tmp)


def replace_inlines_with_xxx(line):
    i = 0
    length = len(line)
    result = []
    
    # stack to keep track of which marker we are "inside" currently.
    marker_stack = []

    # helper to see if the substring at i starts with a marker
    # handle '**', '*', '`', etc.
    def check_marker(s, pos):
        # return (marker_str, marker_length) or (None, 0)
        if s.startswith('```', pos):  # triple backtick inline is rare but let's skip
            return ('```', 3)
        if s.startswith('**', pos):
            return ('**', 2)
        if s.startswith('*', pos):
            return ('*', 1)
        if s.startswith('`', pos):
            return ('`', 1)
        return (None, 0)
    
    # keep track of "plain text" segments outside any markers and replace them with 'xxx'
    plain_text_buffer = []

    def flush_plain_text():
        if plain_text_buffer:
            # check if there's any non-whitespace in the buffer
            buf_content = "".join(plain_text_buffer)
            if buf_content.strip() == "":
                # purely whitespace => keep as is
                result.append(buf_content)
            else:
                # there's some text => replace entire buffer with "xxx"
                result.append("xxx")
            plain_text_buffer.clear()

    while i < length:
        marker, m_len = check_marker(line, i)
        if marker:
            # found an inline marker and  flush any plain text
            flush_plain_text()

            # check a nested marker or open a new one
            if marker_stack and marker_stack[-1] == marker:
                # it's a closing of the current marker
                # output the marker as-is
                result.append(marker)
                marker_stack.pop()
            else:
                # open a new marker
                result.append(marker)
                marker_stack.append(marker)

            i += m_len
        else:
            # not a marker => gather it as plain text
            plain_text_buffer.append(line[i])
            i += 1

    # end of line => flush any remaining text
    flush_plain_text()

    return "".join(result)

def remove_special_characters(args):
    with open(args.input_path, "r") as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        response = data[i][-1]["content"]
        data[i][-1]["content"] = _remove_special_characters(response)

    with open(args.output_path, "w") as f:
        json.dump(data, f)

def shuffle(args):
    if "letter" in args.transform_mode:
        shuffle_func = lambda x: ''.join(random.sample(x.replace(" ", ""), len(x.replace(" ", ""))))
    elif "word" in args.transform_mode:
        shuffle_func = lambda x: ' '.join(random.sample(x.split(), len(x.split())))
    
    with open(args.input_path, "r") as f:
        data = json.load(f)
    
    for i in tqdm(range(len(data))):
        response = data[i][-1]["content"]
        data[i][-1]["content"] = shuffle_func(_remove_special_characters(response))

    with open(args.output_path, "w") as f:
        json.dump(data, f)

def markdown_elements_only(args):
    with open(args.input_path, "r") as f:
        data = json.load(f)
    
    for i in tqdm(range(len(data))):
        response = data[i][-1]["content"]
        data[i][-1]["content"] = remove_text(response)

    with open(args.output_path, "w") as f:
        json.dump(data, f)

def paraphrase(args):
    if args.transform_mode == "paraphrase":
        template = "\"{msg}\"\n\nParaphrase the above text while maintaining the semantic meaning of the original text."
    elif args.transform_mode == "translate":
        template = "\"{msg}\"\n\nTranslate the above text into Chinese."
    elif args.transform_mode == "summarize":
        template = "\"{msg}\"\n\nSummarize the above text in one paragraph."
    
    client = OpenAI(api_key=args.api_key)
    with open(args.input_path, "r") as f:
        data = json.load(f)
    
    for i in tqdm(range(len(data))):
        response = data[i][-1]["content"]
        prompt = template.format(msg=response)
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        data[i][-1]["content"] = completion.choices[0].message.content
        
        # save the transformed responses per generated response
        with open(args.output_path, "w") as f:
            json.dump(data, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None, help="a json path that stores the generated responses from an LLM")
    parser.add_argument("--output_path", type=str, default=None, help="a json path that stores the transformed responses")
    parser.add_argument("--transform_mode", type=str, default=None, 
                        choices=["remove_special_characters", "shuffle_word", 
                                 "shuffle_letter", "markdown_elements_only", 
                                 "paraphrase", "translate", "summarize"],
                        help="the transformation mode to apply to the responses")
    parser.add_argument("--api_key", type=str, default=None, help="the API key for the rewriting model (e.g. GPT-4o-mini)")
    args = parser.parse_args()
    
    random.seed(42)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.transform_mode == "remove_special_characters":
        remove_special_characters(args)
    elif args.transform_mode in ["shuffle_word", "shuffle_letter"]:
        shuffle(args)
    elif args.transform_mode == "markdown_elements_only":
        markdown_elements_only(args)
    elif args.transform_mode in ["paraphrase", "translate", "summarize"]:
        paraphrase(args)

