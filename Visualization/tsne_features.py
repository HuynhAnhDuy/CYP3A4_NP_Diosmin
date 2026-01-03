import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the MACCS keys and labels from CSV files
data = pd.read_csv('CYP3A4_all_rdkit.csv')  # MACCS keys
labels = pd.read_csv('CYP3A4_all_labeled.csv')  # Column with 'Corrosion' and 'Non-corrosion' labels

# Step 2: Apply t-SNE for dimensionality reduction
perplex = 30
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplex, random_state=42)
tsne_results = tsne.fit_transform(data)

# Step 3: Create a DataFrame with t-SNE results and labels
tsne_df = pd.DataFrame(tsne_results, columns=['tSNE1', 'tSNE2'], index=data.index)
tsne_df['Label'] = labels['Label']  # Make sure to use the correct column from your labels file

# Step 4: Visualize the t-SNE results
plt.figure(figsize=(3,3))
sns.scatterplot(x='tSNE1', y='tSNE2', hue='Label', data=tsne_df, 
                palette={'Corrosive': "tomato", 'Noncorrosive': "royalblue"}, 
                alpha=0.5, s=40, edgecolor="white",marker='o')  # alpha for transparency, s for point size

# Add title and customize legend
plt.title('RDKit', fontsize=12, fontweight='bold', fontname ='sans-serif')

# Customize legend position and font size
legend = plt.legend(title='', loc='lower right', bbox_to_anchor=(1.5, 0), fontsize=11)
for text in legend.get_texts():
    text.set_fontsize(11)  # Set legend text size
    text.set_fontweight('bold')
    text.set_fontname('sans-serif')  # Set legend text font to Arial (if available)

# Set axes labels and make them bold
plt.xlabel('tSNE1', fontsize=12, fontweight='bold', fontname ='sans-serif',fontstyle='italic')
plt.ylabel('tSNE2', fontsize=12, fontweight='bold', fontname ='sans-serif',fontstyle='italic')

# Step 5: Save the plot with 300 dpi

plt.savefig('tsne_corrosion_rdkit.svg', dpi=300, bbox_inches='tight')

plt.tight_layout()
