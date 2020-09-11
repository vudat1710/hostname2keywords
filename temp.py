import pandas as pd
import ast

def merge_columns(desc, web_title, desc_lang, web_title_lang):
    if desc_lang in ["en", "vi"]:
        if desc_lang == web_title_lang:
            return desc + ". " + web_title.strip()
    return ""

# df = pd.read_csv("data/merge_crawled_data.csv")
# df["merged_text"] = df.apply(lambda x: merge_columns(x.description, x.web_title, x.desc_lang, x.web_title_lang), axis=1)
# df.to_csv("data/merge_crawled_data.csv", index=False)

df1 = pd.read_csv("res_topics.csv", converters={"en": ast.literal_eval, "vi": ast.literal_eval})
df2 = pd.read_csv("res_topics_web_title.csv", converters={"en": ast.literal_eval, "vi": ast.literal_eval})
df2 = df2.rename(columns={"en": "en_title"})
df2 = df2.rename(columns={"vi": "vi_title"})
df = pd.concat([df1, df2["en_title"], df2["vi_title"]], axis=1)
df["en"] = df["en"] + df["en_title"]
df["vi"] = df["vi"] + df["vi_title"]
df["en"] = df["en"].apply(lambda x: list(set(x)))
df["vi"] = df["vi"].apply(lambda x: list(set(x)))
df = df.drop(["en_title", "vi_title"], axis=1)
df.to_csv("final_keywords.csv", index=False)