import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import pybaseball as pbb
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Set page name and icon
st.set_page_config(
    page_title = 'MLB Predictor',
    page_icon = 'n.png',
)

# Set page title
st.title('Predicting MLB winning probabilities')
st.markdown('## ')

# Initiate sidebar
st.sidebar.markdown('## ENTER PARAMETERS')

# Enter inputs
count = 0

your_team = st.sidebar.selectbox('__Select your team__', (
'ARI',
'ATL',
'BOS',
'CHC',
'CHW',
'CIN',
'CLE',
'COL',
'DET',
'HOU',
'KCR',
'LAA',
'LAD',
'MIA',
'MIL',
'MIN',
'NYM',
'NYY',
'OAK',
'PHI',
'PIT',
'SDP',
'SEA',
'SFG',
'STL',
'TBR',
'TEX',
'TOR',
'WSN'
), index = 16, key = count)

count += 1

your_home = st.sidebar.selectbox('Are they playing at home?', ('Yes','No'), index = 0, key = count)

count += 1

your_pitcher = st.sidebar.text_input('Who is their starting pitcher?', value = 'Carrasco', key = count)

count += 1

your_odds = st.sidebar.number_input('Enter decimal odds from your betting app', value = 1.69, key = count)

count += 1

st.sidebar.markdown('## ')
st.sidebar.markdown('## ')

opp_team = st.sidebar.selectbox('__Select the opposing team__', (
'ARI',
'ATL',
'BOS',
'CHC',
'CHW',
'CIN',
'CLE',
'COL',
'DET',
'HOU',
'KCR',
'LAA',
'LAD',
'MIA',
'MIL',
'MIN',
'NYM',
'NYY',
'OAK',
'PHI',
'PIT',
'SDP',
'SEA',
'SFG',
'STL',
'TBR',
'TEX',
'TOR',
'WSN'
), index = 26, key = count)

count += 1

opp_pitcher = st.sidebar.text_input('Who is their starting pitcher?', value = 'Gray', key = count)

count += 1

opp_odds = st.sidebar.number_input('Enter decimal odds from your betting app', value = 2.25, key = count)

# Configure sidebar
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## Discussion')
st.sidebar.markdown('* One approach to reduce risk when constructing an investment portfolio \
is __maximum diversification portfolio optimization__. In particular, this strategy maximizes \
a diversification ratio and does not take into account expected returns.')
st.sidebar.markdown('* The __diversification ratio__ is defined as the ratio of the weighted average \
of all the volatilites in the portfolio divided by the total portfolio volatility.')
st.sidebar.markdown('* This web app optimizes a portfolio of stocks, bonds, and gold at regular \
intervals.')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')

# Logo in the sidebar
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}" target="_blank">
            <img src="data:image/{img_format};base64,{bin_str}" width = "75" />
        </a>'''
    return html_code

png_html = get_img_with_href('n.png', 'https://www.jimisinith.com')

col1, col2, col3 = st.sidebar.columns([3,7,1])
with col1:
    st.write("")
with col2:
    st.markdown(png_html, unsafe_allow_html=True)
with col3:
    st.write("")


if your_home == 'Yes':
    home_team = your_team
    away_team = opp_team
    home_pitcher = your_pitcher
    away_pitcher = opp_pitcher
    home_decimal_odds = your_odds
    away_decimial_odds = opp_odds
else:
    home_team = opp_team
    away_team = your_team
    home_pitcher = opp_pitcher
    away_pitcher = your_pitcher
    home_decimal_odds = opp_odds
    away_decimal_odss = your_odds


# Main code: copy/paste from my existing code
next_game_home_text_1 = 'Opp_' + away_team
next_game_home_text_2 = 'Opposing_Pitcher_' + away_pitcher
next_game_home_text_3 = 'Starting_Pitcher_' + home_pitcher
next_game_away_text_1 = 'Opp_' + home_team
next_game_away_text_2 = 'Opposing_Pitcher_' + home_pitcher
next_game_away_text_3 = 'Starting_Pitcher_' + away_pitcher



end_date = date.today()
end_season = end_date.year
number_of_seasons = 4
seasons_range = [end_season - i for i in sorted(list(range(0,number_of_seasons)), reverse=True)]

test_size = 0.25

# Home team
schedule = pd.concat([pbb.schedule_and_record(season, home_team) for season in seasons_range], ignore_index=True)
schedule = schedule[schedule['W/L'].notna()]
schedule.loc[schedule['W/L'] == 'L-wo', 'W/L'] = 'L'
schedule.loc[schedule['W/L'] == 'W-wo', 'W/L'] = 'W'
data = schedule[['Home_Away','Opp','Win','Loss','W/L']]


# Define functions
def starting_pitcher(win_loss, win_pitcher, loss_pitcher):
    if win_loss == 'W':
        return win_pitcher
    elif win_loss == 'L':
        return loss_pitcher
    else:
        return 'error'

def opposing_pitcher(win_loss, win_pitcher, loss_pitcher):
    if win_loss == 'W':
        return loss_pitcher
    elif win_loss == 'L':
        return win_pitcher
    else:
        return 'error'

def home(home_away):
    if home_away == '@':
        return 0
    elif home_away == 'Home':
        return 1
    else:
        return 0

def result(win_loss):
    if win_loss == 'W':
        return 1
    elif win_loss == 'L':
        return 0
    else:
        return 0

data['Starting_Pitcher'] = data.apply(lambda row: starting_pitcher(row['W/L'], row['Win'],row['Loss']), axis=1)
data['Opposing_Pitcher'] = data.apply(lambda row: opposing_pitcher(row['W/L'], row['Win'],row['Loss']), axis=1)
data['Home'] = data.apply(lambda row: home(row['Home_Away']), axis=1)
data.drop(['Home_Away','Win','Loss'], axis=1, inplace=True)
data['Result'] = data.apply(lambda row: result(row['W/L']), axis=1)
data.drop('W/L', axis=1, inplace=True)
first_col = data.pop('Result')
data.insert(0, 'Result', first_col)
data = pd.get_dummies(data)


# ### Create 'X' and 'y' split and fit baseline models
X = data.drop('Result', axis=1)
y = data['Result']

models = {
    'Knn'           : KNeighborsClassifier(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'SVM'           : SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes' : GaussianNB(),
    'Gradient Boost' : GradientBoostingClassifier()
}


# Initiate new results table
results_dict = {'Classifier':[],
            'Train Accuracy':[],
            'Test Accuracy':[]
           }

# Train-Test split
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)

    # Loop through models
    for model_name, model in models.items():
        model.fit(X_train, y_train);
        train_accuracy = model.score(X_train, y_train)*100
        test_accuracy = model.score(X_test, y_test)*100
        results_dict['Classifier'].append(model_name)
        results_dict['Train Accuracy'].append(train_accuracy)
        results_dict['Test Accuracy'].append(test_accuracy)
    results_df = pd.DataFrame(results_dict)
    if i % 10 == 0:
        print(str(datetime.now().ctime()) + ': ' + str(i/100*100) + "%")

print(str(datetime.now().ctime()))

summary = results_df[['Classifier','Train Accuracy','Test Accuracy']].groupby(['Classifier']).mean()
summary['Composite Score'] = summary['Train Accuracy'] * 0.25 + summary['Test Accuracy']

for i in range(len(summary)):
    if summary.iloc[i]['Composite Score'] == summary['Composite Score'].max():
        best_model_home = summary.iloc[i].name
        test_accuracy_home = summary.iloc[i]['Test Accuracy']
        print('')
        print('The best model is ' + '\033[1m' + str(best_model_home) + '\033[0m' + ' --> ' + str(models[best_model_home]))

try:

    zero_data = np.zeros(shape=(1,len(X.columns)))
    next_game = pd.DataFrame(zero_data, columns=X.columns)

    # Set relevant columns to 1:
    next_game['Home'] = 1
    next_game[next_game_home_text_1] = 1
    next_game[next_game_home_text_2] = 1
    next_game[next_game_home_text_3] = 1

    next_game = np.array(next_game)
    model = models[best_model_home]
    model.fit(X, y)
    prediction_home = model.predict(next_game.reshape(1, -1))
    prediction_prob_home = model.predict_proba(next_game.reshape(1, -1))

    try:

        zero_data = np.zeros(shape=(1,len(X.columns)))
        next_game = pd.DataFrame(zero_data, columns=X.columns)

        # Set relevant columns to 1:
        next_game['Home'] = 1
        next_game[next_game_home_text_1] = 1
        #next_game[next_game_home_text_2] = 1
        next_game[next_game_home_text_3] = 1

        next_game = np.array(next_game)
        model = models[best_model_home]
        model.fit(X, y)
        prediction_home = model.predict(next_game.reshape(1, -1))
        prediction_prob_home = model.predict_proba(next_game.reshape(1, -1))

        try:

            zero_data = np.zeros(shape=(1,len(X.columns)))
            next_game = pd.DataFrame(zero_data, columns=X.columns)

            # Set relevant columns to 1:
            next_game['Home'] = 1
            next_game[next_game_home_text_1] = 1
            next_game[next_game_home_text_2] = 1
            #next_game[next_game_home_text_3] = 1

            next_game = np.array(next_game)
            model = models[best_model_home]
            model.fit(X, y)
            prediction_home = model.predict(next_game.reshape(1, -1))
            prediction_prob_home = model.predict_proba(next_game.reshape(1, -1))

        except:

            zero_data = np.zeros(shape=(1,len(X.columns)))
            next_game = pd.DataFrame(zero_data, columns=X.columns)

            # Set relevant columns to 1:
            next_game['Home'] = 1
            #next_game[next_game_home_text_1] = 1
            next_game[next_game_home_text_2] = 1
            next_game[next_game_home_text_3] = 1

            next_game = np.array(next_game)
            model = models[best_model_home]
            model.fit(X, y)
            prediction_home = model.predict(next_game.reshape(1, -1))
            prediction_prob_home = model.predict_proba(next_game.reshape(1, -1))

    except:

        print('Missing history')

except:

        print('Missing history')

if prediction_home ==  1:
    print('WIN ~> Probability of winning is: ' + str(prediction_prob_home[0][1]))
else:
    print('LOSE ~> Probability of losing is: ' + str(prediction_prob_home[0][0]))

betway_probability_home = 1 / home_decimal_odds
print(betway_probability_home)



# Away team
schedule = pd.concat([pbb.schedule_and_record(season, away_team) for season in seasons_range], ignore_index=True)
schedule = schedule[schedule['W/L'].notna()]
schedule.loc[schedule['W/L'] == 'L-wo', 'W/L'] = 'L'
schedule.loc[schedule['W/L'] == 'W-wo', 'W/L'] = 'W'

data = schedule[['Home_Away','Opp','Win','Loss','W/L']]


def starting_pitcher(win_loss, win_pitcher, loss_pitcher):
    if win_loss == 'W':
        return win_pitcher
    elif win_loss == 'L':
        return loss_pitcher
    else:
        return 'error'

def opposing_pitcher(win_loss, win_pitcher, loss_pitcher):
    if win_loss == 'W':
        return loss_pitcher
    elif win_loss == 'L':
        return win_pitcher
    else:
        return 'error'


def home(home_away):
    if home_away == '@':
        return 0
    elif home_away == 'Home':
        return 1
    else:
        return 0

def result(win_loss):
    if win_loss == 'W':
        return 1
    elif win_loss == 'L':
        return 0
    else:
        return 0

data['Starting_Pitcher'] = data.apply(lambda row: starting_pitcher(row['W/L'], row['Win'],row['Loss']), axis=1)
data['Opposing_Pitcher'] = data.apply(lambda row: opposing_pitcher(row['W/L'], row['Win'],row['Loss']), axis=1)
data['Home'] = data.apply(lambda row: home(row['Home_Away']), axis=1)
data.drop(['Home_Away','Win','Loss'], axis=1, inplace=True)
data['Result'] = data.apply(lambda row: result(row['W/L']), axis=1)
data.drop('W/L', axis=1, inplace=True)
first_col = data.pop('Result')
data.insert(0, 'Result', first_col)
data = pd.get_dummies(data)


# Create 'X' and 'y' split and fit baseline models
X = data.drop('Result', axis=1)
y = data['Result']

models = {
    'Knn'           : KNeighborsClassifier(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'SVM'           : SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes' : GaussianNB(),
    'Gradient Boost' : GradientBoostingClassifier()
}

# Initiate new results table
results_dict = {'Classifier':[],
            'Train Accuracy':[],
            'Test Accuracy':[]
           }

# Train-Test split
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)

    # Loop through models
    for model_name, model in models.items():
        model.fit(X_train, y_train);
        train_accuracy = model.score(X_train, y_train)*100
        test_accuracy = model.score(X_test, y_test)*100
        results_dict['Classifier'].append(model_name)
        results_dict['Train Accuracy'].append(train_accuracy)
        results_dict['Test Accuracy'].append(test_accuracy)
    results_df = pd.DataFrame(results_dict)
    if i % 10 == 0:
        print(str(datetime.now().ctime()) + ': ' + str(i/100*100) + "%")

print(str(datetime.now().ctime()))

summary = results_df[['Classifier','Train Accuracy','Test Accuracy']].groupby(['Classifier']).mean()
summary['Composite Score'] = summary['Train Accuracy'] * 0.25 + summary['Test Accuracy']

for i in range(len(summary)):
    if summary.iloc[i]['Composite Score'] == summary['Composite Score'].max():
        best_model_away = summary.iloc[i].name
        test_accuracy_away = summary.iloc[i]['Test Accuracy']
        print('')
        print('The best model is ' + '\033[1m' + str(best_model_away) + '\033[0m' + ' --> ' + str(models[best_model_away]))


try:

    zero_data = np.zeros(shape=(1,len(X.columns)))
    next_game = pd.DataFrame(zero_data, columns=X.columns)

    # Set relevant columns to 1:
    next_game['Home'] = 0
    next_game[next_game_away_text_1] = 1
    next_game[next_game_away_text_2] = 1
    next_game[next_game_away_text_3] = 1

    next_game = np.array(next_game)
    model = models[best_model_away]
    model.fit(X, y)
    prediction_away = model.predict(next_game.reshape(1, -1))
    prediction_prob_away = model.predict_proba(next_game.reshape(1, -1))

    try:

        zero_data = np.zeros(shape=(1,len(X.columns)))
        next_game = pd.DataFrame(zero_data, columns=X.columns)

        # Set relevant columns to 1:
        next_game['Home'] = 0
        next_game[next_game_away_text_1] = 1
        #next_game[next_game_away_text_2] = 1
        next_game[next_game_away_text_3] = 1

        next_game = np.array(next_game)
        model = models[best_model_away]
        model.fit(X, y)
        prediction_away = model.predict(next_game.reshape(1, -1))
        prediction_prob_away = model.predict_proba(next_game.reshape(1, -1))

        try:

            zero_data = np.zeros(shape=(1,len(X.columns)))
            next_game = pd.DataFrame(zero_data, columns=X.columns)

            # Set relevant columns to 1:
            next_game['Home'] = 0
            next_game[next_game_away_text_1] = 1
            next_game[next_game_away_text_2] = 1
            #next_game[next_game_away_text_3] = 1

            next_game = np.array(next_game)
            model = models[best_model_away]
            model.fit(X, y)
            prediction_away = model.predict(next_game.reshape(1, -1))
            prediction_prob_away = model.predict_proba(next_game.reshape(1, -1))

        except:

            zero_data = np.zeros(shape=(1,len(X.columns)))
            next_game = pd.DataFrame(zero_data, columns=X.columns)

            # Set relevant columns to 1:
            next_game['Home'] = 0
            #next_game[next_game_away_text_1] = 1
            next_game[next_game_away_text_2] = 1
            next_game[next_game_away_text_3] = 1

            next_game = np.array(next_game)
            model = models[best_model_away]
            model.fit(X, y)
            prediction_away = model.predict(next_game.reshape(1, -1))
            prediction_prob_away = model.predict_proba(next_game.reshape(1, -1))

    except:

        print('Missing history')

except:

        print('Missing history')


if prediction_away ==  1:
    print('WIN ~> Probability of winning is: ' + str(prediction_prob_away[0][1]))
else:
    print('LOSE ~> Probability of losing is: ' + str(prediction_prob_away[0][0]))



betway_probability_away = 1 / away_decimal_odds
print(betway_probability_away)


# Print results

print('')
print('')

if your_team == home_team:
    print('\033[91m' + '\033[1m' + home_team + '\033[0m' + '\033[90m')
    if prediction_home ==  1:
        print('\033[1m' + 'Result: WIN ~> Probability of winning is: ' + str(prediction_prob_home[0][1]) + '\033[0m')
    else:
        print('\033[1m' + 'Result: LOSE ~> Probability of losing is: ' + str(prediction_prob_home[0][0]) + '\033[0m')
    print(str(best_model_home) + ' --> ' + str(models[best_model_home]) + ' --> ' + 'Test accuracy: ' + str(test_accuracy_home))
    print('')
    print('Moneyline decimal odds: ' + str(home_decimal_odds) + ' ~> Win probability: ' + str(betway_probability_home))

if your_team == away_team:
    print('\033[91m' + '\033[1m' + away_team + '\033[0m' + '\033[90m')
    if prediction_away ==  1:
        print('\033[1m' + 'Result: WIN ~> Probability of winning is: ' + str(prediction_prob_away[0][1]) + '\033[0m')
    else:
        print('\033[1m' + 'Result: LOSE ~> Probability of losing is: ' + str(prediction_prob_away[0][0]) + '\033[0m')
    print(str(best_model_away) + ' --> ' + str(models[best_model_away]) + ' --> ' + 'Test accuracy: ' + str(test_accuracy_away))
    print('')
    print('Moneyline decimal odds: ' + str(away_decimal_odds) + ' ~> Win probability: ' + str(betway_probability_away))

print('')
print('')
