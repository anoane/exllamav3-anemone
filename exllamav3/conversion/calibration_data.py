import torch
import os
import random
import json


def split_art(articles, rows, columns, tokenizer):
    t_rows = []
    idx = 0
    empty = torch.empty((1, 0), dtype = torch.long)
    t_row = empty
    while len(t_rows) < rows:
        add_special_tokens = (len(t_rows) % 2 == 0)
        # wrap for high -cr on the stock mix (upstream never pulls >250 rows)
        t_art = tokenizer.encode(articles[idx % len(articles)], add_bos = add_special_tokens, add_eos = add_special_tokens)
        t_row = torch.cat((t_row, t_art), dim = -1)
        t_row = t_row[:, :columns]
        if t_row.shape[-1] == columns:
            t_rows.append(t_row)
            t_row = empty
        idx += 1
    return t_rows


def split_wiki(text, rows, columns, tokenizer):
    articles = [a[a.find("\n") + 1:] for a in text.split("</doc>\n")]
    articles = [a for a in articles if len(a) > 50]
    return split_art(articles, rows, columns, tokenizer)


def split_tiny(text, rows, columns, tokenizer):
    articles = [a.strip() for a in text.split("<|endoftext|>")]
    return split_art(articles, rows, columns, tokenizer)


def shuffle_lines(text, rows, columns, tokenizer):
    articles = text.split("\n")
    articles = [a for a in articles if not a.isspace()]
    random.seed(0)
    random.shuffle(articles)
    return split_art(articles, rows, columns, tokenizer)


def split_raw(text, rows, columns, tokenizer):
    # take at most `rows` FULL cal_cols-wide windows from the file, and NEVER repeat.
    # If the file is shorter than rows*columns, return only the full rows it actually has
    # (fewer than requested) rather than tiling — repeating raws would skew the calibration
    # toward that content. A short file simply contributes fewer rows; total ends up under the
    # requested cal_rows, which is honest and unskewed.
    t_all = tokenizer.encode(text)
    total = t_all.shape[-1]
    avail = total // columns                 # number of full windows available
    n = min(rows, avail)
    t_rows = []
    for i in range(n):
        a = i * columns
        b = a + columns
        t_rows.append(t_all[:, a:b])
    return t_rows


def random_data(text, rows, columns, tokenizer):
    vocab_size = tokenizer.actual_vocab_size
    torch.manual_seed(0)
    t_rows = []
    for i in range(rows):
        t_row = torch.randint(0, vocab_size, (1, columns), dtype = torch.long)
        t_rows.append(t_row)
    return t_rows


def _stock_get_default_calibration(args, tokenizer):
    """Stock exllamav3 mix (wiki/c4/code/…/random). Used only when ANEMONE_CALIB_DIR is unset."""
    columns = args["cal_cols"]
    rows = args["cal_rows"]

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_cal_data")
    files = [
        ("c4.utf8", 20, shuffle_lines),
        ("code.utf8", 20, split_raw),
        ("multilingual.utf8", 10, shuffle_lines),
        ("technical.utf8", 10, split_raw),
        ("wiki.utf8", 50, split_wiki),
        ("tiny.utf8", 5, split_tiny),
        (None, 20, random_data),
    ]

    dist_sum = sum(x for (_, x, _) in files)
    cal_data = []

    for filename, weight, processor in files:
        target_rows = max(1, int(weight / dist_sum * rows))
        if filename:
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding = "utf8") as f:
                file_text = f.read()
        else:
            file_text = None
            target_rows = max(1, rows - len(cal_data))
        r = processor(file_text, target_rows, columns, tokenizer)
        cal_data += r

    return cal_data


def get_default_calibration(args, tokenizer):
    """
    ANEMONE domain calibration.

    If ANEMONE_CALIB_DIR is set, load the domain calibration corpus described by
    <dir>/manifest.json (a code/security taxonomy). Each domain file is split with split_raw
    (contiguous cal_cols-wide token windows, tiled if the file is short) and the number of rows
    is distributed across domains in proportion to their weight_pct (relative; normalized here).
    Otherwise fall back to the stock exllamav3 calibration mix.

    -cr / --cal_rows and -cc / --cal_cols still control the total (default 250x2048); for this
    corpus use -cr 2000 so every domain-specialized expert is exercised (see the routing probe,
    the calibration documentation in the recipe repository).
    """
    columns = args["cal_cols"]
    rows = args["cal_rows"]

    corpus = os.environ.get("ANEMONE_CALIB_DIR")
    if not corpus:
        return _stock_get_default_calibration(args, tokenizer)

    man = json.load(open(os.path.join(corpus, "manifest.json")))
    domains = man["domains"]

    # ANEMONE_CALIB_ALL=1  ->  USE ALL ROWS: take EVERY full cal_cols-wide window from EVERY file,
    # ignoring the weight cap (so no file is truncated). Effective weighting becomes proportional to
    # each file's actual token count. This consumes the entire corpus; --cal_rows is ignored.
    if os.environ.get("ANEMONE_CALIB_ALL") == "1":
        cal = []
        for d in domains:
            path = os.path.join(corpus, d["file"])
            with open(path, "r", encoding="utf8") as f:
                text = f.read()
            got = split_raw(text, 10 ** 9, columns, tokenizer)   # 1e9 target => all avail windows
            print(f" -- ANEMONE calib[ALL]: {d['file']}  {len(got)} rows", flush=True)
            cal += got
        print(f" -- ANEMONE calib[ALL]: {len(cal)} rows / {len(cal)*columns} tokens "
              f"from {len(domains)} files (weight caps ignored)", flush=True)
        return cal

    # weighted mode (default): rows apportioned by weight_pct
    dist = sum(d["weight_pct"] for d in domains)
    cal = []
    for d in domains:
        target = max(1, round(d["weight_pct"] / dist * rows))
        path = os.path.join(corpus, d["file"])
        with open(path, "r", encoding="utf8") as f:
            text = f.read()
        got = split_raw(text, target, columns, tokenizer)   # <= target, never padded/repeated
        if len(got) < target:
            print(f" !! ANEMONE calib: {d['file']} short — {len(got)}/{target} rows "
                  f"(no repeat; contributes fewer)", flush=True)
        cal += got
    print(f" -- ANEMONE calib: {len(cal)} rows from {len(domains)} domains "
          f"(requested {rows})", flush=True)
    return cal

def get_file_calibration(args, tokenizer):
    """
    Calibration rows from a packed token file (safetensors with an "input_ids" tensor of shape
    (rows, cols)), e.g. a self-sampled in-domain trace from sc_trace.py. The file must have been
    produced with the same tokenizer/model family; rows/cols are cropped to the requested
    calibration size.
    """
    from safetensors.torch import load_file
    packed = load_file(args["cal_data"])["input_ids"]
    rows, columns = args["cal_rows"], args["cal_cols"]
    if packed.shape[0] < rows or packed.shape[1] < columns:
        raise ValueError(
            f"Calibration file {args['cal_data']} is {packed.shape[0]} rows x {packed.shape[1]} "
            f"tokens, need {rows} x {columns}"
        )
    if packed.max().item() >= tokenizer.actual_vocab_size:
        raise ValueError(
            f"Calibration file {args['cal_data']} contains token ids outside the model's vocab; "
            f"was it produced with a different tokenizer?"
        )
    return [packed[i : i + 1, :columns].to(torch.long).contiguous() for i in range(rows)]
