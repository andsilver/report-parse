from doctr.io import DocumentFile
from doctr.models import ocr_predictor


model = ocr_predictor(pretrained=True)
# PDF
doc = DocumentFile.from_pdf("data/TEST - Magellan - Cloud Example and Explanation.pdf")
# Analyze
result = model(doc)

# result.show(doc)
json_output = result.export()
# print(json.dumps(json_output, indent=4))

# with open("result/TEST - Magellan - Cloud Example and Explanation.json", "w") as f:
#     json.dump(json_output, f, indent=2)

