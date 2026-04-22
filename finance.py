"""
PROJECT 5: Personal & Business Financial Expense Analysis
Author: Srinath M
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(77)
n = 2500

# ── Generate Dataset ──────────────────────────────────────────
categories  = ['Salary','Food','Transport','Utilities','Entertainment',
               'Healthcare','Shopping','Education','Rent','Savings']
departments = ['Operations','Marketing','HR','IT','Finance']
months      = list(range(1,13))
years       = [2024, 2025, 2026]

txn_type    = np.random.choice(['Income','Expense'], n, p=[0.25,0.75])
amount      = np.where(
    txn_type=='Income',
    np.round(np.random.uniform(30000,150000,n),2),
    np.round(np.random.uniform(500,50000,n),2)
)
category    = np.where(
    txn_type=='Income',
    np.random.choice(['Salary','Freelance','Investment'], n),
    np.random.choice(['Food','Transport','Utilities','Entertainment',
                      'Healthcare','Shopping','Education','Rent'], n)
)
month_num   = np.random.choice(months, n)
year        = np.random.choice(years, n)

df = pd.DataFrame({
    'TransactionID' : [f'TXN{i:05d}' for i in range(1,n+1)],
    'Date'          : [f'{y}-{m:02d}-{np.random.randint(1,29):02d}'
                       for y,m in zip(year,month_num)],
    'Year'          : year,
    'Month'         : month_num,
    'Type'          : txn_type,
    'Category'      : category,
    'Department'    : np.random.choice(departments, n),
    'Amount'        : amount,
    'PaymentMode'   : np.random.choice(['Cash','Card','UPI','NetBanking'], n,
                                        p=[0.15,0.30,0.40,0.15]),
    'IsRecurring'   : np.random.choice([0,1], n, p=[0.60,0.40])
})

df.to_csv('/home/claude/projects/project5_covid/finance_data.csv', index=False)

# ── Analysis ──────────────────────────────────────────────────
print("="*55)
print("  PROJECT 5: FINANCIAL EXPENSE ANALYSIS")
print("="*55)

income_df  = df[df['Type']=='Income']
expense_df = df[df['Type']=='Expense']

total_income  = income_df['Amount'].sum()
total_expense = expense_df['Amount'].sum()
net_savings   = total_income - total_expense

print(f"\nDataset Shape       : {df.shape}")
print(f"Total Transactions  : {len(df)}")
print(f"Total Income        : ₹{total_income:,.0f}")
print(f"Total Expenses      : ₹{total_expense:,.0f}")
print(f"Net Savings         : ₹{net_savings:,.0f}")
print(f"Savings Rate        : {(net_savings/total_income)*100:.1f}%")

print("\nExpense Breakdown by Category:")
cat_exp = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
for cat,val in cat_exp.items():
    pct = val/total_expense*100
    print(f"  {cat:20s}: ₹{val:>12,.0f}  ({pct:.1f}%)")

print("\nYearly Summary:")
yearly = df.groupby(['Year','Type'])['Amount'].sum().unstack(fill_value=0)
print(yearly.to_string())

print("\nPayment Mode Usage:")
print(df['PaymentMode'].value_counts().to_string())

print("\nNumPy Stats on Expenses:")
exp_vals = expense_df['Amount'].values
print(f"  Mean   : ₹{np.mean(exp_vals):,.0f}")
print(f"  Median : ₹{np.median(exp_vals):,.0f}")
print(f"  Std    : ₹{np.std(exp_vals):,.0f}")
print(f"  Max    : ₹{np.max(exp_vals):,.0f}")
print(f"  Min    : ₹{np.min(exp_vals):,.0f}")

# Monthly trend
monthly = df.groupby(['Year','Month','Type'])['Amount'].sum().reset_index()

# ── Visualizations ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Financial Expense Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Income vs Expense (Bar)
ax = axes[0,0]
summary = pd.Series({'Total Income':total_income,'Total Expense':total_expense,'Net Savings':net_savings})
colors_bar = ['#27ae60','#e74c3c','#3498db']
summary.plot(kind='bar', ax=ax, color=colors_bar, edgecolor='black')
ax.set_title('Income vs Expense vs Savings')
ax.set_ylabel('Amount (₹)')
ax.tick_params(axis='x', rotation=30)

# 2. Expense Pie
ax = axes[0,1]
cat_exp.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
ax.set_title('Expense Category Breakdown')
ax.set_ylabel('')

# 3. Monthly Trend
ax = axes[0,2]
for yr in [2024,2025,2026]:
    for tp, lbl, col in [('Income','Income','green'),('Expense','Expense','red')]:
        subset = monthly[(monthly['Year']==yr) & (monthly['Type']==tp)]
        if not subset.empty:
            ax.plot(subset['Month'], subset['Amount'],
                    marker='o', label=f'{yr} {lbl}', alpha=0.7)
ax.set_title('Monthly Income vs Expense Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Amount (₹)')
ax.legend(fontsize=6, ncol=2)
ax.set_xticks(range(1,13))

# 4. Payment Mode Distribution
ax = axes[1,0]
df['PaymentMode'].value_counts().plot(kind='bar', ax=ax, color='#9b59b6', edgecolor='black')
ax.set_title('Payment Mode Usage')
ax.set_ylabel('Number of Transactions')
ax.tick_params(axis='x', rotation=30)

# 5. Yearly grouped bar
ax = axes[1,1]
yearly_plot = df.groupby(['Year','Type'])['Amount'].sum().unstack()
yearly_plot.plot(kind='bar', ax=ax, color=['#e74c3c','#27ae60'], edgecolor='black')
ax.set_title('Year-wise Income vs Expense')
ax.set_ylabel('Amount (₹)')
ax.tick_params(axis='x', rotation=0)
ax.legend(['Expense','Income'])

# 6. Expense Heatmap (Category vs Month)
ax = axes[1,2]
expense_pivot = expense_df.pivot_table(values='Amount', index='Category',
                                        columns='Month', aggfunc='sum', fill_value=0)
sns.heatmap(expense_pivot, cmap='Reds', ax=ax, fmt='.0f')
ax.set_title('Expense Heatmap: Category × Month')
ax.set_xlabel('Month')

plt.tight_layout()
plt.savefig('/home/claude/projects/project5_covid/finance_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Dashboard saved: finance_dashboard.png")
print("✅ Dataset saved : finance_data.csv")