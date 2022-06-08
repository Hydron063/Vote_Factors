from itertools import chain
import pingouin as pg
import pandas as pd
import pyreadstat
import numpy as np
from factor_analyzer import FactorAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
from tabulate import tabulate

df = pd.read_spss('./ZA5960_v1-0-0.sav')
df_temp = pd.read_spss('./ZA5960_v1-0-0.sav', convert_categoricals=False)
l1 = ['v' + str(i) for i in range(1, 56) if i not in (4, 14, 19, 31, 50, 51, 52)]
l2 = ['SEX', 'EMPREL', 'WRKSUP', 'TYPORG2', 'UNION', 'RELIGGRP', 'URBRURAL', 'COUNTRY', 'DEGREE', 'PARTY_LR']
num = l1 + ['DEGREE']
df[num] = df_temp[num]
del df_temp

print('Features before:', df.columns.values)
df = df.drop(list(set(df.columns.values) ^ set(l1 + l2)), axis=1)
print('Features after:', df.columns.values)
df = df.dropna(axis=0).reset_index(drop=True)
df[num] = df[num].astype(int)

cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.astype(str))

df = df.loc[df['COUNTRY'].isin(['AT-Austria', 'BG-Bulgaria', 'CZ-Czech Republic', 'DE-Germany', 'HU-Hungary',
                                'IE-Ireland', 'LV-Latvia', 'NL-Netherlands', 'PL-Poland', 'SK-Slovak Republic',
                                'SI-Slovenia', 'ES-Spain', 'SE-Sweden', 'DK-Denmark', 'FI-Finland', 'FR-France',
                                'PT-Portugal'])]
print('Remaining countries:', df['COUNTRY'].unique())

# Recoding and dummies
four = ['v' + str(i) for i in chain(range(1, 13), range(20, 30), range(49, 50)) if i != 4]
five = ['v' + str(i) for i in list(range(13, 19)) + list(range(30, 39)) + [39, 42, 44, 47, 54, 55] if i not in (14, 31)]
df[four] = df[four].apply(lambda x: x.apply(lambda z: 5 - int(z)))
df[five] = df[five].apply(lambda x: x.apply(lambda z: 6 - int(z)))

dummies = ['v41', 'v53', 'SEX', 'EMPREL', 'WRKSUP', 'TYPORG2', 'UNION', 'RELIGGRP', 'URBRURAL', 'COUNTRY']
df = pd.get_dummies(df, columns=dummies, prefix=dummies)
for s in [',', '/', '(', ')']:
    df.columns = df.columns.str.replace(s, '', regex=True)
df.columns = df.columns.str.replace(' ', '_', regex=True)
df.columns = df.columns.str.replace('-', '_', regex=True)
df.columns = df.columns.str.replace('A_country_village_farm_or_home_in_the_country', 'Country_or_village', regex=True)
df.columns = df.columns.str.replace('The_suburbs_or_outskirts_of_a_big_city_a_town_or_a_small_city', 'The_suburbs',
                                    regex=True)
df = df.drop(
    ['RELIGGRP_No_religion', 'SEX_Female', 'WRKSUP_No', 'TYPORG2_Public_employer', 'UNION_No_member', 'v41_2', 'v53_2',
     'URBRURAL_Country_or_village', 'COUNTRY_CZ_Czech_Republic', 'EMPREL_Working_for_own_family\'s_business'], axis=1)

# Exploratory analysis
fa = FactorAnalyzer(rotation='oblimin', method='ml')
f_names = list(set(l1) ^ {'v36', 'v37', 'v38', 'v41', 'v53', 'v54', 'v55'})
factors = df[f_names]
fa.fit(factors)
ev, v = fa.get_eigenvalues()
ev_perc = [ev_i/sum(ev)*100 for ev_i in ev]
print('Percentage of variance explained:', *[str(i)+'%,' for i in np.round(ev_perc, 2)])
print('Cumulative variance for selected features:', [round(sum(ev_perc[:i+1])) for i in range(len(ev))])
plt.title('Scree plot')
plt.xlabel('Factors number')
plt.ylabel('Eigenvalue')
plt.xticks(range(factors.shape[1]))
plt.yticks(range(int(max(ev))+1))
plt.plot(range(1, factors.shape[1] + 1), ev)
plt.show()

# Confirmatory analysis
fa = FactorAnalyzer(n_factors=5, rotation='oblimin', method='ml')
new_variables = fa.fit_transform(factors)
loads = pd.DataFrame(fa.loadings_)
headers = ['Variable name', 'ETHN', 'POLIT', 'ANTI', 'BLIND', 'CULT']
table = tabulate(np.c_[np.array(sorted(f_names)), np.round(fa.loadings_, 3)], headers, tablefmt="fancy_grid")
print(table)
f = open('structure_matrix.txt', 'w', encoding='utf8')
f.write(table)

ethn = ['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v8', 'v9', 'v11', 'v12', 'v13', 'v49']
polit = ['v10', 'v16', 'v20', 'v21', 'v22', 'v23', 'v29']
anti = ['v' + str(i) for i in range(39, 49) if i != 41]
cult = ['v18', 'v24', 'v25', 'v26', 'v27', 'v28']
blind = ['v15', 'v17', 'v30', 'v32', 'v33', 'v34', 'v35']
ethn_alpha = pg.cronbach_alpha(df[ethn])
polit_alpha = pg.cronbach_alpha(df[polit])
anti_alpha = pg.cronbach_alpha(df[anti])
cult_alpha = pg.cronbach_alpha(df[cult])
blind_alpha = pg.cronbach_alpha(df[blind])
print('Cronbach\'s alpha for every feature:', ethn_alpha, polit_alpha, anti_alpha, cult_alpha, blind_alpha)
df = df.drop(ethn + anti + polit + blind + cult, axis=1)
df[headers[1:]] = new_variables

# Changing features order
df = df[['ANTI', 'POLIT', 'CULT', 'ETHN', 'BLIND', 'v36', 'v37', 'v38', 'v41_1', 'v53_1', 'v54', 'v55',
         'UNION_Member', 'TYPORG2_Private_employer', 'WRKSUP_Yes', 'EMPREL_Employee', 'URBRURAL_The_suburbs',
         'URBRURAL_A_big_city', 'SEX_Male', 'DEGREE', 'RELIGGRP_Buddhist', 'RELIGGRP_Catholic',
         'RELIGGRP_Islamic', 'RELIGGRP_Jewish', 'RELIGGRP_Orthodox', 'RELIGGRP_Other_Christian',
         'RELIGGRP_Other_Religions', 'RELIGGRP_Protestant', 'COUNTRY_AT_Austria', 'COUNTRY_DK_Denmark',
         'COUNTRY_ES_Spain', 'COUNTRY_FI_Finland', 'COUNTRY_FR_France', 'COUNTRY_HU_Hungary', 'COUNTRY_IE_Ireland',
         'COUNTRY_LV_Latvia', 'COUNTRY_NL_Netherlands', 'COUNTRY_PL_Poland', 'COUNTRY_PT_Portugal',
         'COUNTRY_SI_Slovenia', 'COUNTRY_SK_Slovak_Republic', 'PARTY_LR']]
df = df[(df.PARTY_LR != 'No party, no preference') & (df.PARTY_LR != 'NAP, no answer in VOTE_LE, not eligible')].\
    reset_index(drop=True)

del_l = []
for fn in df.columns:
    for d_i in dummies:
        if d_i in fn:
            t = df[fn].astype(int).sum()
            if t < 100:
                del_l.append(fn)
df = df.drop(del_l, axis=1)
print('Deleted features:', del_l)

print('Total number of respondents:', len(df))
gener_c = np.unique(df['PARTY_LR'], return_counts=True)
print('Dependent variable distribution:', gener_c[0], gener_c[1] / len(df))
df.groupby('PARTY_LR').ANTI.count().to_frame('Respondents').plot.bar(rot=0)
plt.xticks(range(len(gener_c[0])), gener_c[0])
plt.xlabel('Electoral preferences')
plt.show()

y = df['PARTY_LR'].to_numpy()
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X = df.drop('PARTY_LR', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

# Tree cross-validation
model = DecisionTreeClassifier(max_depth=20, min_samples_leaf=20, min_samples_split=40)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='f1_weighted', cv=cv, n_jobs=-1)
print('F1 per iteration', n_scores)
print('Mean F1: %.3f' % (np.mean(n_scores)))

# Tree test F1 and feature importance
res = model.fit(X_train, y_train).predict(X_test)
print('Model prediction distribution:', np.unique(res, return_counts=True))
print('Test F1:', f1_score(y_test, res, average='weighted'))
print('Depth of tree and number of leaves:', model.get_depth(), model.get_n_leaves())
headers = ['Feature name', 'Feature importance']
table = tabulate(np.c_[model.feature_names_in_, np.round(model.feature_importances_, 3)], headers, tablefmt="fancy_grid")
f = open('tree.txt', 'w', encoding='utf8')
f.write(table)

# Representaion
text_representation = tree.export_text(model)
print(text_representation)
plt.figure()
plot_tree(model, filled=True, rounded=True, impurity=False, fontsize=3)
plt.title("Decision tree trained on all features")
plt.show()

# Random Forest
model = RandomForestClassifier()
space = {
    'bootstrap': [True],
    'max_depth': [None, 80, 120],
    'max_features': [5, 'auto'],
    'min_samples_leaf': [1, 5, 10],
    'min_samples_split': [2, 10, 20],
    'n_estimators': [100, 200, 500]
}
search = GridSearchCV(model, space, scoring='f1_macro', n_jobs=-1, cv=cv)
result = search.fit(X_train, y_train)
score, params = result.best_score_, result.best_params_

# score, params = 0.368, {'bootstrap': True, 'max_depth': 80, 'max_features': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

print('Sklearn mean F1: %.3f' % score)
print('Best params:', params)

clf = RandomForestClassifier(random_state=42, **params)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Model prediction percent distribution:', np.unique(pred, return_counts=True)[1] / len(pred))
print('Test f1 score: %.3f' % f1_score(y_test, pred, average='weighted'))
print('Accuracy: %.3f' % accuracy_score(y_test, pred))
headers = ['Feature name', 'Feature importance']
table = tabulate(np.c_[clf.feature_names_in_, np.round(clf.feature_importances_, 3)], headers, tablefmt="fancy_grid")
f = open('random_forest.txt', 'w', encoding='utf8')
f.write(table)

# Base model
coefs = np.unique(y, return_counts=True)[1] / len(y)
print('Data percent distribution:', coefs)
f1_base = 1 / 3 * np.sum(coefs ** 2 / (coefs + 1 / 6))
print('Base F1: %.3f' % f1_base)
print('Base accuracy: %.3f' % np.sum(coefs/6))

# Random forest representation
print('Number of trees:', len(clf.estimators_))
text_representation = tree.export_text(clf.estimators_[0])
print(text_representation)
plt.figure()
plot_tree(clf.estimators_[0], filled=True, rounded=True, impurity=False, fontsize=3)
plt.title("First tree of the forest")
plt.show()
