

import pandas as pd




df = pd.read_csv('predicted_plan_div3.csv')

df.rename(columns= {'Predicted Plan' : 'predicted_plan'},inplace=True)


busbus = 0
busper = 0
buspre = 0
perbus = 0
perper = 0
perpre = 0
prebus = 0
preper = 0
prepre = 0


for index, row in df.iterrows():
    
    plan = row.plan_purchased_nice
    predicted = row.predicted_plan
    if plan == "Business":
        if predicted == plan:
            busbus += 1
        elif predicted == "Personal":
            busper += 1
        else:
            buspre += 1
    elif plan == "Personal":
        if predicted == plan:
            perper += 1
        elif predicted == "Premium":
            perpre += 1
        else:
            perbus += 1
    else:
        if predicted == plan:
            prepre += 1
        elif predicted == "Business":
            prebus += 1
        else:
            preper += 1
            
print('busbus:', busbus, 'busper: ', busper, "buspre", buspre, "perbus", perbus, "perper", perper, "perpre", perpre, "prebus", prebus, "preper", preper, "prepre", prepre)
            
            


p = busbus + perper + prepre
n = busper + buspre + perbus + perpre + prebus +preper

accuracy_div_three = (p)/(p+n)

accuracy_div_three





df = pd.read_csv('predicted_plan_div4.csv')

df.rename(columns= {'Predicted Plan' : 'predicted_plan'},inplace=True)

busbus = 0
busper = 0
buspre = 0
perbus = 0
perper = 0
perpre = 0
prebus = 0
preper = 0
prepre = 0


for index, row in df.iterrows():
    
    plan = row.plan_purchased_nice
    predicted = row.predicted_plan
    if plan == "Business":
        if predicted == plan:
            busbus += 1
        elif predicted == "Personal":
            busper += 1
        else:
            buspre += 1
    elif plan == "Personal":
        if predicted == plan:
            perper += 1
        elif predicted == "Premium":
            perpre += 1
        else:
            perbus += 1
    else:
        if predicted == plan:
            prepre += 1
        elif predicted == "Business":
            prebus += 1
        else:
            preper += 1
            
print('busbus:', busbus, 'busper: ', busper, "buspre", buspre, "perbus", perbus, "perper", perper, "perpre", perpre, "prebus", prebus, "preper", preper, "prepre", prepre)
            
            
p = busbus + perper + prepre
n = busper + buspre + perbus + perpre + prebus +preper

accuracy_div_four = (p)/(p+n)

accuracy_div_four




df = pd.read_csv('predicted_plan_div5.csv')

df.rename(columns= {'Predicted Plan' : 'predicted_plan'},inplace=True)

busbus = 0
busper = 0
buspre = 0
perbus = 0
perper = 0
perpre = 0
prebus = 0
preper = 0
prepre = 0


for index, row in df.iterrows():
    
    plan = row.plan_purchased_nice
    predicted = row.predicted_plan
    if plan == "Business":
        if predicted == plan:
            busbus += 1
        elif predicted == "Personal":
            busper += 1
        else:
            buspre += 1
    elif plan == "Personal":
        if predicted == plan:
            perper += 1
        elif predicted == "Premium":
            perpre += 1
        else:
            perbus += 1
    else:
        if predicted == plan:
            prepre += 1
        elif predicted == "Business":
            prebus += 1
        else:
            preper += 1
            
print('busbus:', busbus, 'busper: ', busper, "buspre", buspre, "perbus", perbus, "perper", perper, "perpre", perpre, "prebus", prebus, "preper", preper, "prepre", prepre)
            
            
p = busbus + perper + prepre
n = busper + buspre + perbus + perpre + prebus +preper

accuracy_div_five = (p)/(p+n)

accuracy_div_five




df = pd.read_csv('predicted_plan_div6.csv')

df.rename(columns= {'Predicted Plan' : 'predicted_plan'},inplace=True)

busbus = 0
busper = 0
buspre = 0
perbus = 0
perper = 0
perpre = 0
prebus = 0
preper = 0
prepre = 0


for index, row in df.iterrows():
    
    plan = row.plan_purchased_nice
    predicted = row.predicted_plan
    if plan == "Business":
        if predicted == plan:
            busbus += 1
        elif predicted == "Personal":
            busper += 1
        else:
            buspre += 1
    elif plan == "Personal":
        if predicted == plan:
            perper += 1
        elif predicted == "Premium":
            perpre += 1
        else:
            perbus += 1
    else:
        if predicted == plan:
            prepre += 1
        elif predicted == "Business":
            prebus += 1
        else:
            preper += 1
            
print('busbus:', busbus, 'busper: ', busper, "buspre", buspre, "perbus", perbus, "perper", perper, "perpre", perpre, "prebus", prebus, "preper", preper, "prepre", prepre)
            
            
p = busbus + perper + prepre
n = busper + buspre + perbus + perpre + prebus +preper

accuracy_div_six = (p)/(p+n)

accuracy_div_six




print('div3:', accuracy_div_three, 'div4:', accuracy_div_four, 'div5:', accuracy_div_five, 'div6:', accuracy_div_six)






