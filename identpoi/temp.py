df = pd.DataFrame.from_dict(data_dict, orient='index')
df[['salary']] = df[['salary']].apply(pd.to_numeric, errors='coerce')
df[['deferral_payments']] = df[['deferral_payments']].apply(pd.to_numeric, errors='coerce')
df[['total_payments']] = df[['total_payments']].apply(pd.to_numeric, errors='coerce')
df[['restricted_stock_deferred']] = df[['restricted_stock_deferred']].apply(pd.to_numeric, errors='coerce')
df[['exercised_stock_options']] = df[['exercised_stock_options']].apply(pd.to_numeric, errors='coerce')
df[['long_term_incentive']] = df[['long_term_incentive']].apply(pd.to_numeric, errors='coerce')
df[['bonus']] = df[['salary']].apply(pd.to_numeric, errors='coerce')
df[['total_stock_value']] = df[['total_stock_value']].apply(pd.to_numeric, errors='coerce')

df.describe()
df.info()

ax1 = df.plot.scatter(x='salary', y='total_payments', c='deferral_payments')

missing_data = []
for key_name in features_list:
    for person_name in data_dict.keys():
        if data_dict[person_name][key_name] == "NaN":
            missing_data.append(person_name + "_" + key_name)

print(missing_data)
