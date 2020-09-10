import pandas as pd
import re
from tldextract import extract

def preprocess_df(df):
    df["count"] = df.groupby(["msisdn", "domain_name"])["domain_name"].transform("count")
    temp = df[df["count"] > 10]
    dn = set(temp["domain_name"])
    dn = [x for x in dn if "dn" not in dn]
    dn = [x.replace("content", "") if "content" in x else x for x in dn]
    dn = [x for x in dn if "static" not in x]
    dn = [x.replace("cdn", "") if "cdn" in x else x for x in dn]
    dn = [x for x in dn if "ad" not in x]
    dn = [x for x in dn if not x.isnumeric()]
    dn = [x for x in dn if "content" not in x]
    dn = [x for x in dn if not re.match("\d+\.\d+\.\d+\.\d+", x)]
    dn = [x.replace("apis", "") if x.endswith("apis") else x for x in dn]
    dn = [x for x in dn if len(x) > 1]
    return list(set(dn))

def filter_url(url):
    sub, dm, suf = extract(url)
    return dm

if __name__=="__main__":
    df = pd.read_csv("super_rich_full_network.tsv", sep="\t", index_col=0)
    dn = preprocess_df(df)