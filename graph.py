import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the CSV file into a DataFrame
file_path = 'datasets/rewards.csv'
df = pd.read_csv(file_path)

# Assuming the columns are named 'rounds', 'data1', 'data2', 'data3', 'data4', 'data5'
rounds = df['Round']
data_columns = ['Reward DDQN', 'Reward Dueling DDQN', 'Reward DQN', 'Reward Dueling DQN', 'Reward D4PG']

# Define a list of colors for the plots using seaborn color palette
palette = sns.color_palette("husl", len(data_columns))

# Initialize the seaborn style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(12, 8))

for i, column in enumerate(data_columns):
    data = df[column]
    sns.lineplot(x=rounds, y=data, label=column, color=palette[i])
    
    # Fit polynomial trend lines
    degree = 3  # Degree of the polynomial
    rounds_poly = np.linspace(rounds.min(), rounds.max(), 100)
    coefficients = np.polyfit(rounds, data, degree)
    trendline_data = np.polyval(coefficients, rounds_poly)
    plt.plot(rounds_poly, trendline_data, linestyle='--', color=palette[i])

# Add titles and labels
plt.title('Reward vs Rounds', fontsize=16)
plt.xlabel('Rounds', fontsize=14)
plt.ylabel('Reward', fontsize=14)

# Add a legend
plt.legend(title='Reward Type')

# Show the plot
plt.show()
