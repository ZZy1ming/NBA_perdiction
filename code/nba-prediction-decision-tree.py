import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load original team data
team_regularseason = pd.read_csv('/Users/apple/Desktop/365 final project/nba_team_stats_00_to_23.csv')
games = pd.read_csv('/Users/apple/Desktop/365 final project/game.csv')

# Data cleaning function
def clean_missing_values(df):
    threshold = 0.3
    df_cleaned = df.drop(columns=df.columns[df.isnull().mean() > threshold])
    
    # Fill missing values
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']):
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    return df_cleaned

nba_team_stats_cleaned = clean_missing_values(team_regularseason)

# Filter recent seasons
recent_seasons = ['2019-20', '2020-21', '2021-22', '2022-23']
recent_stats = nba_team_stats_cleaned[nba_team_stats_cleaned['season'].isin(recent_seasons)].copy()

# Calculate basic statistics
def calculate_team_statistics(df):
    df['avg_points_per_game'] = df['points'] / df['games_played']
    df['avg_points_allowed'] = (df['field_goals_attempted'] - df['field_goals_made']) / df['games_played']
    df['net_points'] = df['avg_points_per_game'] - df['avg_points_allowed']
    df['win_percentage'] = df['wins'] / df['games_played']
    df['avg_rebounds'] = df['rebounds'] / df['games_played']
    df['avg_assists'] = df['assists'] / df['games_played']
    return df

recent_stats = calculate_team_statistics(recent_stats)

# Calculate rolling averages
def rolling_average_features(df, n=5):
    df = df.sort_values(by=['Team', 'season', 'games_played'])
    rolling_cols = ['points', 'rebounds', 'assists', 'wins']
    for col in rolling_cols:
        df[f'recent_{col}_avg'] = df.groupby('Team')[col].transform(lambda x: x.rolling(window=n, min_periods=1).mean())
    return df

recent_stats = rolling_average_features(recent_stats)

# Calculate efficiency features
def calculate_efficiency_features(df):
    df['pace'] = df['field_goals_attempted'] / df['games_played']
    df['offensive_rating'] = df['points'] / df['field_goals_attempted'] * 100
    df['defensive_rating'] = (df['field_goals_attempted'] - df['field_goals_made']) / df['games_played'] * 100
    df['net_rating'] = df['offensive_rating'] - df['defensive_rating']
    return df

recent_stats = calculate_efficiency_features(recent_stats)

# Generate match features
def generate_match_features(df, team1, team2):
    team1_data = df[df['Team'] == team1].mean(numeric_only=True)
    team2_data = df[df['Team'] == team2].mean(numeric_only=True)
    
    if team1_data.isnull().all() or team2_data.isnull().all():
        return None

    match_features = {
        'team1_win_percentage': team1_data['win_percentage'],
        'team2_win_percentage': team2_data['win_percentage'],
        'win_percentage_diff': team1_data['win_percentage'] - team2_data['win_percentage'],
        'team1_net_points': team1_data['net_points'],
        'team2_net_points': team2_data['net_points'],
        'net_points_diff': team1_data['net_points'] - team2_data['net_points'],
        'team1_avg_rebounds': team1_data['avg_rebounds'],
        'team2_avg_rebounds': team2_data['avg_rebounds'],
        'rebounds_diff': team1_data['avg_rebounds'] - team2_data['avg_rebounds'],
        'team1_avg_assists': team1_data['avg_assists'],
        'team2_avg_assists': team2_data['avg_assists'],
        'assists_diff': team1_data['avg_assists'] - team2_data['avg_assists'],
        'team1_offensive_rating': team1_data['offensive_rating'],
        'team2_offensive_rating': team2_data['offensive_rating'],
        'offensive_rating_diff': team1_data['offensive_rating'] - team2_data['offensive_rating'],
        'team1_defensive_rating': team1_data['defensive_rating'],
        'team2_defensive_rating': team2_data['defensive_rating'],
        'defensive_rating_diff': team1_data['defensive_rating'] - team2_data['defensive_rating'],
    }
    return match_features

# Create match dataset
def create_match_dataset(df, games_df):
    if not np.issubdtype(games_df['game_date'].dtype, np.datetime64):
        games_df['game_date'] = pd.to_datetime(games_df['game_date'])

    start_date = pd.to_datetime('2019-02-14 00:00:00')
    end_date = pd.to_datetime('2023-02-19 00:00:00')
    filtered_games = games_df[(games_df['game_date'] >= start_date) & (games_df['game_date'] <= end_date)]
    
    match_data = []
    match_labels = []
    features_example = None

    for idx, row in filtered_games.iterrows():
        home_team = row['team_name_home']
        away_team = row['team_name_away']
        print(f"Processing match: {home_team} vs {away_team}")
        
        features = generate_match_features(df, home_team, away_team)
        if features is None:
            print(f"Missing data for match: {home_team} vs {away_team}")
            continue
        
        match_data.append(list(features.values()))
        if features_example is None:
            features_example = features
        label = 1 if row['wl_home'] == 'W' else 0
        match_labels.append(label)

    if not match_data:
        print("No valid matches found. Returning empty dataset.")
        return pd.DataFrame(), np.array([])
    
    match_data_df = pd.DataFrame(match_data, columns=features_example.keys())
    return match_data_df, np.array(match_labels)

# Create dataset
match_dataset, match_labels = create_match_dataset(recent_stats, games)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(match_dataset, match_labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train Decision Tree model
dt_classifier = DecisionTreeClassifier(
    max_depth=5,  # Prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature importance visualization
feature_importance = pd.DataFrame({
    'feature': match_dataset.columns,
    'importance': dt_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Most Important Features for Game Prediction')
plt.tight_layout()
plt.show()

# Example prediction
team1 = 'Boston Celtics'
team2 = 'Denver Nuggets'
new_match_features = generate_match_features(recent_stats, team1, team2)
if new_match_features is not None:
    input_features = scaler.transform([list(new_match_features.values())])
    prediction = dt_classifier.predict(input_features)
    prediction_prob = dt_classifier.predict_proba(input_features)
    print(f"\nPredicted winner (1=home win): {prediction[0]}")
    print(f"Win probability: {prediction_prob[0][1]:.2f}")
