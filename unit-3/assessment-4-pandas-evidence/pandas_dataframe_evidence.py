import pandas as pd

# Create a Pandas DataFrame from a dictionary.
og_dataframe = pd.DataFrame({
    "Task": ["Notebook", "Screenshots", "GitHub Upload"],
    "Library": ["Pandas", "Pandas", "Pandas"],
    "Status": ["Created", "Captured", "Uploaded"],
    "Score": [82, 76, 91]
})
print("Created DataFrame:")
print(og_dataframe)

# Access a column, a row, and one cell.
print("Status column:")
print(og_dataframe["Status"])
print("Second row:")
print(og_dataframe.iloc[1])
print("First task status:", og_dataframe.loc[0, "Status"])

# Delete a column and a row using drop().
print("DataFrame after deleting Score column:")
print(og_dataframe.drop("Score", axis=1))
print("DataFrame after deleting row index 1:")
print(og_dataframe.drop(1, axis=0))
