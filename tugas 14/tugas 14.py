import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

# Data transaksi
transactions = [
    ['M', 'O', 'N', 'K', 'E', 'Y'],
    ['D', 'O', 'N', 'K', 'E', 'Y'],
    ['M', 'A', 'K', 'E'],
    ['M', 'U', 'C', 'K', 'Y'],
    ['C', 'O', 'O', 'K', 'E']]

# Encoding ke bentuk biner
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori dengan min_support = 0.6
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Association Rules dengan min_confidence = 0.8
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules = rules.dropna()
rules = rules[rules['confidence'] != float('inf')]

# Tampilkan itemset dan rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Visualisasi dengan NetworkX
def draw_graph(rules_df):
    G = nx.DiGraph()  
    for _, row in rules_df.iterrows():
        for ant in row['antecedents']:
            for cons in row['consequents']:
                G.add_edge(ant, cons, weight=row['confidence'])
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=2)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Visualisasi Aturan Asosiasi")
    plt.show()

draw_graph(rules)