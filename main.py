from doctr.io import DocumentFile
import numpy as np
from fuzzywuzzy import fuzz
from doctr.models import ocr_predictor
import json
import pandas as pd


class ParseReport:
    def __init__(self, file, json_file=False, save_json=False):
        if json_file:
            df = pd.read_json(file)
        else:
            model = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_pdf(file)
            result = model(doc)
            json_output = result.export()
            df = pd.DataFrame(json_output)

            if save_json:
                with open(save_json, "w") as f:
                    json.dump(json_output, f, indent=2)

        pages = df.join(pd.json_normalize(df.pop('pages')))

        blocks = pages.explode("blocks")
        blocks['block_idx'] = np.arange(blocks.shape[0])
        blocks['index'] = blocks['block_idx']
        blocks = blocks.set_index('index')

        blocks = blocks.join(pd.json_normalize(blocks.pop('blocks')))
        blocks = blocks.rename(columns={'geometry': 'block_geometry'})

        lines = blocks.explode("lines")
        lines['line_idx'] = np.arange(lines.shape[0])
        lines['index'] = np.arange(lines.shape[0])
        lines = lines.set_index('index')

        lines = lines.join(pd.json_normalize(lines.pop('lines')))
        lines = lines.rename(columns={'geometry': 'line_geometry'})

        words = lines.explode("words")
        words['word_idx'] = np.arange(words.shape[0])
        words['index'] = np.arange(words.shape[0])
        words = words.set_index('index')

        words = words.join(pd.json_normalize(words.pop('words')))
        words = words.rename(columns={'geometry': 'word_geometry'})

        words["word_geometry"] = words.word_geometry.apply(lambda x: {"x1": x[0][0], "y1": x[0][1], "x2": x[1][0], "y2": x[1][1]})

        self.words = words.join(pd.json_normalize(words.pop('word_geometry')))

    def find_attribute_one(self, word, extract_value=True, context=None):
        if context is None:
            context = self.words
        try:
            line = context[context['value'].str.contains(word)][["page_idx", "block_idx", "line_idx", "value"]].values.squeeze()
            line = context[(context['page_idx'] == line[0]) & (context['block_idx'] == line[1]) & (context['line_idx'] == line[2])]["value"].values
            if extract_value:
                return self.extract_value(" ".join(line))

            return " ".join(line)
        except Exception as e:
            print(e)
            return None

    def find_attribute_two(self, word1, word2, extract_value=True, context=None):
        if context is None:
            context = self.words
        try:
            line = context[context['value'].str.contains(word1) | context['value'].str.contains(word2)][
                ["page_idx", "block_idx", "line_idx", "x1", "y1", "x2", "y2", "value"]]
            line['paired'] = line.value + " " + line.value.shift(-1)
            line['space'] = line.x1.shift(-1) - line.x2
            line['align'] = ((line.y1.shift(-1) - line.y1) + (line.y2.shift(-1) - line.y2)) / 2

            line["score"] = line["paired"].apply(lambda x: 100 - fuzz.ratio(str(x), str(word1) + " " + str(word2)))
            line = line[line.score < 10]
            line = line.sort_values(by=['score', 'space', "align"])
            line = line.iloc[0][["page_idx", "block_idx", "line_idx", "paired"]]
            line = context[(context['page_idx'] == line.page_idx.item()) & (context['block_idx'] == line.block_idx.item()) & (
                        context['line_idx'] == line.line_idx.item())]["value"].values
            if extract_value:
                return self.extract_value(" ".join(line))

            return " ".join(line)
        except Exception as e:
            print(e)
            return None

    def company_name(self):
        return " ".join(self.words[(self.words['page_idx'] == 0) & (self.words['block_idx'] == 0) & (self.words['line_idx'] == 0)]["value"].values)

    @staticmethod
    def extract_value(sentence):
        return sentence.split(":")[-1].strip()

    def get_lines(self, word1, word2, context=None):
        if context is None:
            context = self.words

        line = context[context['value'].str.contains(word1) | context['value'].str.contains(word2)][
            ["page_idx", "block_idx", "line_idx", "x1", "y1", "x2", "y2", "value"]]
        line['paired'] = line.value + " " + line.value.shift(-1)
        line['space'] = line.x1.shift(-1) - line.x2
        line['align'] = ((line.y1.shift(-1) - line.y1) + (line.y2.shift(-1) - line.y2)) / 2

        line["score"] = line["paired"].apply(lambda x: 100 - fuzz.ratio(str(x), str(word1) + " " + str(word2)))
        line = line[line.score < 10]
        line = line.sort_values(by=['score', 'space', "align"])

        return line

    def get_lien(self):
        line = self.get_lines("Lien", "Type:")

        lien_list = []
        for i, row in line.iterrows():
            lien = self.words[(self.words.page_idx == row.page_idx) & (self.words.block_idx == row.block_idx)]

            lien_dict = {
                "Lien Type:": self.find_attribute_two("Lien", "Type:", context=lien),
                "Filed Against:": self.find_attribute_two("Filed", "Against:", context=lien),
                "Amount:": self.find_attribute_one("Amount:", context=lien),
                "Recorded Date:": self.find_attribute_two("Recorded", "Date:", context=lien),
                "Recording Information:": self.find_attribute_two("Recording", "Information:", context=lien),
                "Comment:": self.find_attribute_one("Comment:", context=lien),
            }
            none = True
            for k, v in lien_dict.items():
                if v is not None:
                    none = False

            if not none:
                lien_list.append(lien_dict)

        return lien_list

    def get_vesting_instrument(self):
        line = self.get_lines("Vesting", "Instrument")

        lien_list = []
        for i, row in line.iterrows():
            lien = self.words[(self.words.page_idx == row.page_idx) & (self.words.block_idx == row.block_idx)]

            lien_dict = {
                "Vesting Instrument Type": self.find_attribute_two("Vesting", "Instrument", context=lien),
                "Executed": self.find_attribute_one("Executed:", context=lien),
                "Recorded": self.find_attribute_one("Recorded:", context=lien),
                "Recording Information": self.find_attribute_two("Recording", "Information:", context=lien),
                "Comment": self.find_attribute_one("Comment:", context=lien),
            }
            none = True
            for k, v in lien_dict.items():
                if v is not None:
                    none = False

            if not none:
                lien_list.append(lien_dict)

        return lien_list

    def get_instrument(self):
        line = self.get_lines("Instrument", "Type:")

        lien_list = []
        for i, row in line.iterrows():
            lien = self.words[(self.words.page_idx == row.page_idx) & (self.words.block_idx == row.block_idx)]

            lien_dict = {
                "Instrument Type:": self.find_attribute_two("Instrument", "Type:", context=lien),
                "From:": self.find_attribute_one("From:", context=lien),
                "To:": self.find_attribute_one("To:", context=lien),
                "Executed:": self.find_attribute_one("Executed:", context=lien),
                "Recorded:": self.find_attribute_one("Recorded:", context=lien),
                "Recording Information:": self.find_attribute_two("Mortgage", "Recording", context=lien),
            }
            none = True
            for k, v in lien_dict.items():
                if v is not None:
                    none = False

            if not none:
                lien_list.append(lien_dict)

        return lien_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='ParseReport',
        description='It parses report',
        epilog='Thanks')

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-j', '--json', required=False, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-a', '--save_json_path', required=False, default=False)
    args = parser.parse_args()
    print("Extracting information for file:", args.file)

    report = ParseReport(file=args.file, json_file=args.json, save_json=args.save_json_path)

    information = {
        "Company Name": report.company_name(),
        "Certification Date": report.find_attribute_two("Certification", "Date"),
        "Search Type": report.find_attribute_two("Search", "Type"),
        "County": report.find_attribute_one("County:"),

        "Client File Number": report.find_attribute_two("File", "Number:"),
        "Property Address": report.find_attribute_two("Property", "Address:"),
        "Brief Legal Description": report.find_attribute_two("Legal", "Description:"),
        "State Parcel ID": report.find_attribute_two("State", "ID:"),
        "Alternate Parcel ID": report.find_attribute_two("Alternate", "ID:"),

        "State Tax Parcel ID": report.find_attribute_two("State", "Tax"),
        "Alternate Tax Parcel ID": report.find_attribute_two("Alternate", "Tax"),
        "Tax Year": report.find_attribute_two("Tax", "Year:"),
        "Land Value": report.find_attribute_two("Land", "Value:"),
        "Improvement Value": report.find_attribute_two("Improvement", "Value:"),
        "Exemption Total": report.find_attribute_two("Exemption", "Total:"),
        "Net Value": report.find_attribute_two("Net", "Value:"),
        "Installment Amount (two annual)": report.find_attribute_two("Installment", "Amount"),
        "Status": report.find_attribute_one("Status"),
        "Purchaser(s)": report.find_attribute_one("Purchaser(s):"),
        "Record Owner(s)": report.find_attribute_two("Record", "Owner(s):"),
        "Vesting Instrument": report.get_vesting_instrument(),
        "Instrument": report.get_instrument(),
        "lien": report.get_lien(),

        "Restricted Real Estate:": report.find_attribute_two("Real", "Estate:"),
        "Recording Information:": report.find_attribute_two("Record", " Owner(s):"),

        "Owner/Grantee:": report.find_attribute_one("Owner/Grantee:"),
        "Year Acquired:": report.find_attribute_two("Year", "Acquired:"),
        "Vesting Instrument Recording Information:": report.find_attribute_two("Instrument", "Recording"),
    }

    # more_information = {
    #
    #     "Vesting Instrument Type:": report.find_attribute_two("Vesting", "Instrument"),
    #     "Executed:": report.find_attribute_one("Executed:"),
    #     "Recorded:": report.find_attribute_one("Recorded:"),
    #     "Recording Information:": report.find_attribute_two("Recording", "Information:"),
    #     "Comment:": report.find_attribute_one("Comment:"),
    #
    #     "Instrument Type:": report.find_attribute_two("Instrument", "Type:"),
    #     "From:": report.find_attribute_one("From:"),
    #     "To:": report.find_attribute_one("To:"),
    #     "Mortgage Executed:": report.find_attribute_two("Mortgage", "Executed:"),
    #     "Mortgage Recorded:": report.find_attribute_two("Mortgage", "Recorded:"),
    #     "Mortgage Recording Information:": report.find_attribute_two("Mortgage", "Recording"),
    #
    #     "Lien Type:": report.find_attribute_two("Lien", "Type:"),
    #     "Filed Against:": report.find_attribute_two("Filed", "Against:"),
    #     # "Amount:": report.find_attribute_two("Record", " Owner(s):"),
    #     "Recorded Date:": report.find_attribute_two("Recorded", "Date:"),
    #     # "Recording Information:": report.find_attribute_two("Record", " Owner(s):"),
    #     # "Comment:": report.find_attribute_two("Record", " Owner(s):"),
    #
    #     # "Lien Type:": report.find_attribute_two("Record", " Owner(s):"),
    #     # "Filed Against:": report.find_attribute_two("Record", " Owner(s):"),
    #     # "Amount:": report.find_attribute_two("Record", " Owner(s):"),
    #     # "Recorded Date:": report.find_attribute_two("Record", " Owner(s):"),
    #     # "Recording Information:": report.find_attribute_two("Record", " Owner(s):"),
    #     # "Comment:": report.find_attribute_two("Record", " Owner(s):"),
    #
    #     "Restricted Real Estate:": report.find_attribute_two("Real", "Estate:"),
    #     # "Recording Information:": report.find_attribute_two("Record", " Owner(s):"),
    #
    #     "Owner/Grantee:": report.find_attribute_one("Owner/Grantee:"),
    #     "Year Acquired:": report.find_attribute_two("Year", "Acquired:"),
    #     "Vesting Instrument Recording Information:": report.find_attribute_two("Instrument", "Recording"),
    # }
    print(json.dumps(information, indent=4))

