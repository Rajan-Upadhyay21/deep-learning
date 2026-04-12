import re

raw_output = "   this is    an AI generated answer. it has extra   spaces and inconsistent casing.   "

cleaned_output = re.sub(r"\s+", " ", raw_output).strip()
cleaned_output = cleaned_output[0].upper() + cleaned_output[1:]

print("Raw Output:")
print(raw_output)

print("\nPostprocessed Output:")
print(cleaned_output)
