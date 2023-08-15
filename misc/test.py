import zstandard
import json
import io

with open(r"C:\Users\fujin\Downloads\reddit_zst\reddit\submissions\RS_2010-08.zst", 'rb') as fh:
    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
    reader = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    for line in text_stream:
        data = json.loads(line)
        print(data)
