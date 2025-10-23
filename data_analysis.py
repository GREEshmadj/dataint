import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


sentiment = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data.csv")


sentiment['date'] = pd.to_datetime(sentiment['date'])
sentiment = sentiment.rename(columns={
    'value': 'fear_greed_value',
    'classification': 'sentiment_label'
})
sentiment = sentiment[['date', 'fear_greed_value', 'sentiment_label']]


trades.columns = trades.columns.str.strip().str.lower().str.replace(' ', '_')
trades['timestamp_ist'] = pd.to_datetime(trades['timestamp_ist'], format="%d-%m-%Y %H:%M", errors='coerce')
trades['trade_date'] = trades['timestamp_ist'].dt.date
trades['trade_date'] = pd.to_datetime(trades['trade_date'])


num_cols = ['execution_price', 'size_tokens', 'size_usd', 'closed_pnl', 'fee']
for c in num_cols:
    trades[c] = pd.to_numeric(trades[c], errors='coerce')

trades['profitable'] = (trades['closed_pnl'] > 0).astype(int)

if 'leverage' not in trades.columns:
    trades['leverage'] = trades['size_usd'] / trades['execution_price'].replace(0, np.nan)


daily_perf = trades.groupby('trade_date').agg(
    total_trades=('account', 'count'),
    avg_pnl=('closed_pnl', 'mean'),
    median_pnl=('closed_pnl', 'median'),
    win_rate=('profitable', 'mean'),
    avg_size_usd=('size_usd', 'mean'),
    total_volume_usd=('size_usd', 'sum')
).reset_index()


merged = pd.merge(daily_perf, sentiment, left_on='trade_date', right_on='date', how='left')
merged.drop(columns=['date'], inplace=True)


print("\n=== Basic Descriptive Statistics ===")
print(merged.describe())

print("\n=== Sentiment Distribution ===")
print(merged['sentiment_label'].value_counts())


corr = merged[['fear_greed_value', 'avg_pnl', 'win_rate', 'total_volume_usd']].corr()
print("\n=== Correlation Matrix ===")
print(corr)

fear_pnl = merged[merged['sentiment_label'].str.contains("Fear", case=False, na=False)]['avg_pnl'].dropna()
greed_pnl = merged[merged['sentiment_label'].str.contains("Greed", case=False, na=False)]['avg_pnl'].dropna()

t_stat, p_val = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
print(f"\nT-test Fear vs Greed on avg_pnl: t={t_stat:.3f}, p={p_val:.3f}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=merged, x='sentiment_label', y='avg_pnl', order=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
plt.title("Average Trader PnL vs Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Daily PnL (USD)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("avg_pnl_vs_sentiment.png", dpi=150)
plt.show()

merged.to_csv("merged_sentiment_trader_performance.csv", index=False)
corr.to_csv("correlation_matrix.csv")

print("\n✅ Analysis complete! Files saved:")
print("- merged_sentiment_trader_performance.csv")
print("- correlation_matrix.csv")
print("- avg_pnl_vs_sentiment.png")

avg_summary = merged.groupby('sentiment_label')[['avg_pnl', 'win_rate', 'total_volume_usd']].mean().sort_values('avg_pnl', ascending=False)
print("\n=== Average Performance by Sentiment ===")
print(avg_summary)
