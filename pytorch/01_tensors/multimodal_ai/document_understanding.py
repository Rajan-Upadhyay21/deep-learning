document = {
    "title": "Invoice",
    "fields": {
        "invoice_number": "INV-1024",
        "date": "2026-04-14",
        "total": "$425.00"
    },
    "layout_regions": ["header", "table", "footer"]
}

print("Document Type:")
print(document["title"])

print("\nExtracted Fields:")
for key, value in document["fields"].items():
    print(f"{key}: {value}")

print("\nLayout Regions:")
print(document["layout_regions"])
