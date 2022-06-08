import pandas as pd
import pyreadstat
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from itertools import chain
from factor_analyzer import FactorAnalyzer
import pingouin as pg
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression


# Sklearn wrapper for statsmodels
class SMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit(maxiter=10, cov_type="hc0")
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return np.argmax(self.results_.predict(X, transform=False).to_numpy(), axis=1)


df = pd.read_spss('./ZA5960_v1-0-0_4.sav')
df_temp = pd.read_spss('./ZA5960_v1-0-0_4.sav', convert_categoricals=False)
l1 = ['v' + str(i) for i in range(1, 56) if i not in (4, 14, 19, 31, 50, 51, 52)]
l2 = ['SEX', 'EMPREL', 'WRKSUP', 'TYPORG2', 'UNION', 'RELIGGRP', 'URBRURAL', 'COUNTRY', 'DEGREE', 'PARTY_LR']
num = l1 + ['DEGREE']
df[num] = df_temp[num]
del df_temp

print('Features before:', df.columns.values)
df = df.drop(list(set(df.columns.values) ^ set(l1 + l2)), axis=1)
print('Features after:', df.columns.values)
df = df.dropna(axis=0).reset_index(drop=True)
df['PARTY_LR'] = df['PARTY_LR'].astype(str)
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

print('Atttention', np.unique(df['PARTY_LR']))

# Exploratory analysis
fa = FactorAnalyzer(rotation='oblimin', method='ml')
f_names = list(set(l1) ^ {'v36', 'v37', 'v38', 'v41', 'v53', 'v54', 'v55'})
factors = df[f_names]
fa.fit(factors)
ev, v = fa.get_eigenvalues()
ev_perc = [ev_i / sum(ev) * 100 for ev_i in ev]
print('Percentage of variance explained:', *[str(i) + '%,' for i in np.round(ev_perc, 2)])
print('Cumulative variance for selected features:', [round(sum(ev_perc[:i + 1])) for i in range(len(ev))])
plt.title('Scree plot')
plt.xlabel('Factors number')
plt.ylabel('Eigenvalue')
plt.xticks(range(factors.shape[1]))
plt.yticks(range(int(max(ev)) + 1))
plt.plot(range(1, factors.shape[1] + 1), ev)
plt.show()

# Confirmatory analysis
fa = FactorAnalyzer(n_factors=5, rotation='oblimin', method='ml')
new_variables = fa.fit_transform(factors)
loads = pd.DataFrame(fa.loadings_)
headers = ['Variable name', 'ETHN', 'POLIT', 'ANTI', 'BLIND', 'CULT']
table = tabulate(np.c_[np.array(f_names), np.round(fa.loadings_, 3)], headers, tablefmt="fancy_grid")
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
df = df[(df.PARTY_LR != 'No party, no preference') & (df.PARTY_LR != 'NAP, no answer in VOTE_LE, not eligible')]. \
    reset_index(drop=True)
df['PARTY_LR'] = df['PARTY_LR'].replace('Center, liberal', '1Center, liberal')

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
X = df.drop('PARTY_LR', axis=1).astype(float)
X2 = add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state=40)

# Multicollinearity test
vif_data = pd.DataFrame()
vif_data['Feature name'] = X_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
headers = ['Feature name', 'VIF']
table = tabulate(np.c_[vif_data['Feature name'], np.round(vif_data['VIF'], 3)], headers, tablefmt="fancy_grid")
print(table)
f = open('vif.txt', 'w', encoding='utf8')
f.write(table)

# statsmodels
interm = sm.MNLogit(y_train, X_train)
est = interm.fit(maxiter=10, cov_type="hc0")
pred = est.predict(X_test, transform=False).to_numpy()
pred = np.argmax(pred, axis=1)
trans = {'Would not/ did not vote': 5, '1Center, liberal': 0, 'Far left (communist etc.)': 1,
         'Far right (fascist etc.)': 2, 'Left, center left': 3, 'Right, conservative': 4}
y_test = np.array([trans[i] for i in y_test])
y_train = np.array([trans[i] for i in y_train])
f = open('multinomial.txt', 'w')
f.write(str(est.summary()))
print(np.unique(pred, return_counts=True)[1] / len(pred))

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=40)
n_scores = cross_val_score(SMWrapper(sm.MNLogit), X_train, y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
print('Statsmodels mean F1: %.3f' % np.mean(n_scores))
print('Statsmodels test F1: %.3f' % f1_score(y_test, pred, average='weighted'))
print('Statsmodels accuracy: %.3f' % accuracy_score(y_test, pred))

# sklearn
model = LogisticRegression()
space = {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'penalty': ['none', 'l1', 'l2', 'elasticnet'],
         'C': [10 ** i for i in range(-5, 3)]}
search = GridSearchCV(model, space, scoring='f1_macro', n_jobs=-1, cv=cv)
result = search.fit(X_train, y_train)
score, params = result.best_score_, result.best_params_

# score, params = 0.457, {'C': 1e-05, 'penalty': 'none', 'solver': 'newton-cg'}

print('Sklearn mean F1: %.3f' % score)
print('Best params:', params)
clf = LogisticRegression(random_state=42, max_iter=1000, **params)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Sklearn test F1: %.3f' % f1_score(y_test, pred, average='weighted'))
print('Sklearn accuracy: %.3f' % accuracy_score(y_test, pred))

# Base model
coefs = np.unique(y, return_counts=True)[1] / len(y)
print('Prediction class distribution:', coefs)
f1_base = 1 / 3 * np.sum(coefs ** 2 / (coefs + 1 / 6))
print('Base model F1: %.3f' % f1_base)
