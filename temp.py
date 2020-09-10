import pandas as pd
import ast

def merge_columns(desc, web_title, desc_lang, web_title_lang):
    if desc_lang in ["en", "vi"]:
        if desc_lang == web_title_lang:
            return desc + ". " + web_title.strip()
    return ""

df = pd.read_csv("data/merge_crawled_data.csv")
df["merged_text"] = df.apply(lambda x: merge_columns(x.description, x.web_title, x.desc_lang, x.web_title_lang), axis=1)
df.to_csv("data/merge_crawled_data.csv", index=False)