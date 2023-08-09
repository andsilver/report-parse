from doctr.io import DocumentFile
import numpy as np
from fuzzywuzzy import fuzz
from doctr.models import ocr_predictor
import json
import pandas as pd
import sys


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


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
            return None

    def find_attribute_two(self, word1, word2, extract_value=True, context=None):
        if context is None:
            context = self.words
        try:
            line = context[context['value'].str.contains(word1) | context['value'].str.contains(word2)][
                ["page_idx", "block_idx", "line_idx", "x1", "y1", "x2", "y2", "value"]]
            line['paired'] = line.value + " " + line.value.shift(-1)
            line['space'] = abs(line.x1.shift(-1) - line.x2)
            line['align'] = (abs(line.y1.shift(-1) - line.y1) + abs(line.y2.shift(-1) - line.y2)) / 2

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
        line['space'] = abs(line.x1.shift(-1) - line.x2)
        line['align'] = (abs(line.y1.shift(-1) - line.y1) + abs(line.y2.shift(-1) - line.y2)) / 2

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

    def find_table_pages(self, word1, word2):
        try:
            context = self.words
            # word1, word2 = "Notice", "Lien"
            line = context[context['value'].str.contains(word1) | context['value'].str.contains(word2)][
                ["page_idx", "block_idx", "line_idx", "x1", "y1", "x2", "y2", "value"]]

            line['paired'] = line.value + " " + line.value.shift(-1)
            line['space'] = line.x1.shift(-1) - line.x2
            line['align'] = (abs(line.y1.shift(-1) - line.y1) + (line.y2.shift(-1) - line.y2)) / 2

            line["score"] = line["paired"].apply(lambda x: 100 - fuzz.ratio(str(x), str(word1) + " " + str(word2)))
            line = line[line.score < 10]
            line = line.sort_values(by=['score', 'space', "align"])

            return line
        except Exception as e:
            return None

    @staticmethod
    def find_column_values(context, word1, word2, right=0.0, left=0.0, height=0.21):
        try:
            # word1, word2 = "Kind", "Tax"
            c = report.words[(report.words.page_idx == context.page_idx)]
            column = c[c['value'].str.contains(word1) | c['value'].str.contains(word2)][
                ["page_idx", "block_idx", "line_idx", "x1", "y1", "x2", "y2", "value"]]
            column['paired'] = column.value + " " + column.value.shift(-1)
            column['space'] = abs(column.x1.shift(-1) - column.x2)
            column['align'] = (abs(column.y1.shift(-1) - column.y1) + (column.y2.shift(-1) - column.y2)) / 2

            column['x12'] = column.x1.shift(-1)
            column['y12'] = column.y1.shift(-1)
            column['x22'] = column.x2.shift(-1)
            column['y22'] = column.y2.shift(-1)

            column["score"] = column["paired"].apply(lambda x: 100 - fuzz.ratio(str(x), str(word1) + " " + str(word2)))
            column = column[column.score < 10]
            column = column.sort_values(by=['score', 'space', "align"])

            column_data = c[(c.x1 >= column.x1.item() - left) & (c.x2 <= column.x22.item() + right) & (c.y1 - 0.01 >= column.y2.item())]
            column_data['hd'] = abs(column_data.y1.shift(-1) - column.y2.item())
            column_data = column_data.sort_values(by=['hd'])
            column_data['bid'] = abs(column_data.block_idx.shift(-1) - column_data.block_idx)
            column_data['lid'] = abs(column_data.line_idx.shift(-1) - column_data.line_idx)
            column_data['h2d'] = abs(column_data.hd.shift(-1) - column_data.hd)
            column_data = column_data.sort_values(by=['hd', 'h2d', 'bid', "lid"])
            column_data = column_data.reset_index(drop=True)
            values = []
            for i, k in column_data.iterrows():
                if context.block_idx == k.block_idx:
                    continue
                # elif k.bid > 1 or k.lid > block_d or column_data.iloc[i + line_d].hd.item() > height:
                #     break
                if k.hd > height or column_data.iloc[i + 1].h2d.item() > height:
                    break
                else:
                    values.append(k)

            return values
        except Exception as e:
            return None

    def get_lien_tables(self):
        notice_lien = self.find_table_pages("Notice", "Lien")
        lien_tables = []
        for i, table in notice_lien.iterrows():
            # table = notice_lien.iloc[0][["page_idx", "block_idx", "line_idx", "paired"]]
            c = self.words[(self.words.page_idx == table.page_idx)]
            kind_of_tax = self.find_column_values(table, "Kind", "Tax")
            first_col = pd.DataFrame(kind_of_tax)
            rows = [["Kind of Tax (a)", "Tax Period Ending (b)", "Identifying Number (c)", "Date of Assessment",  "Last Day for Refining (e)", "Unpaid Balance of Assessment (f)"]]
            for k, v in first_col.iterrows():
                line_values = list(c[(c.y1 >= c[c.line_idx == v.line_idx].y1.min()) & (c.y2 <= c[c.line_idx == v.line_idx].y2.max())].value.values)
                line_values.append(line_values[0])
                rows.append(line_values[1:])

            if len(rows) > 1:
                lien_tables.append(rows)

        return lien_tables


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
        "Recording Information:": report.find_attribute_two("Record", "Owner(s):"),

        "Owner/Grantee:": report.find_attribute_one("Owner/Grantee:"),
        "Year Acquired:": report.find_attribute_two("Year", "Acquired:"),
        "Vesting Instrument Recording Information:": report.find_attribute_two("Instrument", "Recording"),
        "federal_tax_lien": report.get_lien_tables()
    }

    print(json.dumps(information, indent=4))
