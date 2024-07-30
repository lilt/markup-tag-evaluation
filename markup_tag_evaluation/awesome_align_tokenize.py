import argparse
from io import TextIOWrapper
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer


def write_lines(f: TextIOWrapper, texts: list[str]):
    for l in texts:
        f.write(l)
        f.write("\n")


if __name__ == "__main__":
    """Script to tokenize source and target files using the tokenizer that awesome align uses, 
    in order to get subword alignments later.
    
    Note: In order to not get [UNK] tokens, need to modify a couple lines in the awesome align code, see:
    - https://github.com/neulab/awesome-align/blob/5f150d45bbe51e167daf0a84abebaeb07c3323d1/awesome_align/tokenization_bert.py#L516
    - https://github.com/neulab/awesome-align/blob/5f150d45bbe51e167daf0a84abebaeb07c3323d1/awesome_align/tokenization_bert.py#L540
    Which can be replaced with `output_tokens.append(token)`
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("source")
    argparser.add_argument("target")
    argparser.add_argument("src_token")
    argparser.add_argument("tgt_token")
    argparser.add_argument("all_token")
    args = argparser.parse_args()

    with open(args.source, 'r') as f:
        sources = [l.strip() for l in f.readlines()]

    with open(args.target, 'r') as f:
        targets = [l.strip() for l in f.readlines()]

    assert len(sources) == len(targets)

    tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    all_tokenized: list[str] = []
    src_tokenized: list[str] = []
    tgt_tokenized: list[str] = []
    for src, tgt in zip(sources, targets):
        token_src = tokenizer.tokenize(src)
        token_tgt = tokenizer.tokenize(tgt)

        src_sub = " ".join(token_src).replace(" ##", " ")
        tgt_sub = " ".join(token_tgt).replace(" ##", " ")
        src_tokenized.append(src_sub)
        tgt_tokenized.append(tgt_sub)
        all_tokenized.append(f"{src_sub} ||| {tgt_sub}")

    with open(args.all_token, 'w') as f:
        write_lines(f, all_tokenized)

    with open(args.src_token, 'w') as f:
        write_lines(f, src_tokenized)

    with open(args.tgt_token, 'w') as f:
        write_lines(f, tgt_tokenized)
