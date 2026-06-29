import pandas as pd

# Create a Pandas Series with labelled elements.
og_series = pd.Series([82, 76, 91], index=["Planning", "Build", "Testing"], name="OG Website Scores")
print("Created Series:")
print(og_series)

# Access elements by label and by position.
print("Planning score by label:", og_series.loc["Planning"])
print("First score by position:", og_series.iloc[0])

# Delete an element by label using drop().
series_after_delete = og_series.drop("Testing")
print("Series after deleting Testing:")
print(series_after_delete)
