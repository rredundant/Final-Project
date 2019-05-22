

import pandas as pd
import glob
import random
from sklearn.utils import shuffle




files = glob.glob('/Users/ehamilton/Downloads/Data/*.csv')


li = []

# for filename in files:
#     num_lines = sum(1 for l in open(filename))
#     size = int(num_lines / 6 ) # use these values: 3,4,5,6
#     skip_idx = random.sample(range(1, num_lines), num_lines - size)
#     df = pd.read_csv(filename, skiprows=skip_idx, index_col=None, header=0)
#     li.append(df)

# df = pd.concat(li, axis=0, ignore_index=True)

# # Shuffle the data.
# df = shuffle(df)
# df.head()

for filename in files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    
df = pd.concat(li, axis=0, ignore_index=True)




# df_drop = df[['plan_purchased_nice', 'msg_whole']]
df.fillna(0, inplace=True)
# df_drop.head()




df.iloc[444].msg_whole




# __domain_primary__ . or 

for_df = []

for index, row in df.iterrows():
    non_zero_tags = []
    plan_purchased_nice = row.plan_purchased_nice
    non_zero_tags.append(plan_purchased_nice)
    non_zero_tags = [x for x in non_zero_tags if x] # if x not zero
    
    if non_zero_tags:
        for_df.append({
            "tags": non_zero_tags,
            "text": row.msg_whole
        })
    else:
        for_df.append({
        "tags": ['notag'],
        "text": row.msg_whole
    })




from sklearn.utils import shuffle
# __label__sauce __label__cheese How much does potato starch affect a cheese sauce recipe?

out = pd.DataFrame.from_records(for_df)
out['tags'] = out['tags'].map(
    
        lambda x:  " , ".join([ "__label__{}".format(j) for j in x])

)
out = shuffle(out)
out.head()




out['for_fast'] = out.tags + " " + out['text']
out['for_fast'].to_clipboard(index=False)




out.head()

len(out)




# magic commands time
# you are here! /Users/ehamilton/fastTextCLI/fastText-0.2.0
# original file is tickets.txt

head -n 60000 tickets.txt > tickets.train
tail -n 20000 tickets.txt > tickets.valid

./fasttext supervised -input tickets.train -output tickets_model

./fasttext test tickets_model.bin tickets.valid

N	739
P@1	0.286
R@1	0.189




from fastText import load_model




m = load_model('tickets_model.bin')




m.predict(df.iloc[555].msg_whole, k=3)
df["Predicted Plan"] = df["msg_whole"].apply(lambda x: m.predict(x)[0][0].split("__")[-1])

df.head()

df.to_csv('predicted_plan_div6.csv')




df.head()




avg [i[1][0] for i in list(df["Predicted Plan"])]

# [i[0][0].split("__")[-1] for i in list(df["Predicted Plan"])]




if 0:
    print('this will not print')




if 1:
    print('this will print')




