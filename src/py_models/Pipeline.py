import datetime
import logging
import sys
from calendar import monthrange
from collections import defaultdict
import re
import random

from faker import Faker

from src.py_models.POSPredictor import POSPredictor
from src.py_models.CorefResolver import CorefResolver
from src.vp_ta_constants import COUNTRIES, SLOVENIAN_MUNICIPALITIES, SLOVENIAN_CITIES, MONTHS, EVENT_NAMES
from src.py_models.NERPredictor import NERPredictor
import classla
import rstr
import nltk
# from dateparser.search import search_dates
import dateparser
import string
# import datefinder


# from nltk.stem import WordNetLemmatizer

class Pipeline:
    def __init__(self, text, mode="readable"):
        self.text: str = text
        self.phone_mapper: defaultdict = defaultdict(str)
        self.name_mapper:  defaultdict = defaultdict(str)
        self.mail_mapper: defaultdict = defaultdict(str)
        self.org_mapper:  defaultdict = defaultdict(str)
        self.address_mapper:  defaultdict = defaultdict(str)
        self.emso_mapper: defaultdict = defaultdict(str)
        self.vat_mapper: defaultdict = defaultdict(str)
        self.misc_mapper: defaultdict = defaultdict(str)
        self.person_mapper: defaultdict = defaultdict(str)
        self.date_mapper: defaultdict = defaultdict(str)
        self.plate_mapper: defaultdict = defaultdict(str)
        self.event_mapper: defaultdict = defaultdict(str)
        self.bank_info_mapper: dict = {"swift": defaultdict(str), "iban": defaultdict(str), "cc_number": defaultdict(str)}
        self.personal_info_mapper: dict = {"passport_no": defaultdict(str), "pid": defaultdict(str)}
        self.classla = classla.Pipeline('sl', processors='tokenize,pos,lemma,ner')
        self.ner_predictor = NERPredictor(text, "EMBEDDIA/sloberta", "../../models/ner/custom_ner_model_sloberta—combined—5e—8b—500ws—5e-05lr_new_test")
        self.pos_predictor = POSPredictor(text, "EMBEDDIA/sloberta", "../../models/xpos/custom_xpos_model_sloberta—combined_all—8e—24b—500ws—5e-05lr_new_test")
        self.coref_resolver = CorefResolver(self.text, "../../models/coref/spanbert-senticoref10")
        Faker.seed(64)
        self.mode = mode
        self.faker: Faker = Faker("sl_SI")
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrainL1OStrategy')
        # classla.download('sl', processors=processors)


    def start_predictions(self, text, mode="readable"):

        self.reset_mappers()

        self.text = text
        self.mode = mode

        self.text = self.coref_resolver.resolve_allennlp(text)
        self.ner_predictor.predict(self.text)
        self.pos_predictor.predict(self.text)
        print(self.coref_resolver.corefs)

        self.replace_organizations()
        self.replace_dates()
        self.replace_misc()
        self.replace_address()
        self.replace_mail()
        self.replace_person()
        self.replace_bank_info()
        self.replace_emsho()
        self.replace_licence_plate()
        self.replace_personal_info()
        self.replace_phones()
        self.replace_events()

        return self.text


    def _calc_new_val(self, placeholder, mapper):
        new_val = ""
        if self.mode == "medium":
            new_val = f"[{placeholder}]"
        elif self.mode == "low":
            new_val = f"[{placeholder}_{len(mapper.keys())}]"
        elif self.mode == "high":
            new_val = "[REDACTED]"
        return new_val

    def replace_phones(self):
        phone_regex = r"\b((0[1-7][0-9])[ \-\/]?([0-9]{3})[ \-\/]?([0-9]{3}))\b|\b([\+]?(386)[ \-\/]?([0-9]{2})[ \-\/]?([0-9]{3})[ \-\/]?([0-9]{3}))\b"
        for phone in re.finditer(phone_regex, self.text, re.MULTILINE):
            pp  = phone.group(0)
            if pp not in self.phone_mapper:
                pp = pp.replace(" ", "")
                pp = pp.replace("/", "")
                pp = pp.replace("-", "")
                if not pp.startswith("+") and len(pp) > 9:
                    continue
                if pp.startswith("+") and len(pp) > 12:
                    continue
                calculated = self._calc_new_val("TELEFON", self.phone_mapper)
                new_val = calculated if calculated != "" else self.faker.phone_number()
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.phone_mapper[pp] = new_val
        for key, val in self.phone_mapper.items():
            self.text = self.text.replace(key, val)
        print(self.text)

    def replace_mail(self):
        email_regex = r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
        comp_addresses = ["info", "kontakt", "kontaktiraj", "prodaja"]
        free_domains = ["gmail.com", "yahoo.com", "live.com", "hotmail.com"]
        for email in re.finditer(email_regex, self.text, re.MULTILINE):
            pp  = email.group(0)
            parts = pp.split("@")
            if pp not in self.mail_mapper:
                new_val = self.faker.ascii_company_email()
                if parts[0] in comp_addresses:
                    new_val = f"{parts[0]}{self.faker.domain_name()}"
                elif parts[1] in free_domains:
                    new_val = self.faker.free_email()
                calculated = self._calc_new_val("EMAIL", self.mail_mapper)
                new_val = calculated if calculated != "" else new_val
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.mail_mapper[pp] = new_val
        for key, val in self.mail_mapper.items():
            self.text = self.text.replace(key, val)

    def replace_address(self):
        locations = self.ner_predictor.tag_dist["loc"]
        print(locations)
        for location in locations:
            lower_loc = self.classla(location.lower()).get("lemma")[0]
            if location not in self.address_mapper and lower_loc not in COUNTRIES and lower_loc not in SLOVENIAN_MUNICIPALITIES and lower_loc not in SLOVENIAN_CITIES:
                self.address_mapper[location] = self.faker.street_name()
            elif location not in self.address_mapper and lower_loc in SLOVENIAN_MUNICIPALITIES:
                self.address_mapper[location] = self.faker.administrative_unit()
            elif location not in self.address_mapper and lower_loc in SLOVENIAN_CITIES:
                self.address_mapper[location] = self.faker.city()
            calculated = self._calc_new_val("ADRESA", self.address_mapper)
            new_val = calculated if calculated != "" else self.address_mapper[location]
            if location[-1] in string.punctuation:
                new_val = f"{new_val}{location[-1]}"
            self.address_mapper[location] = new_val

        for key, value in self.address_mapper.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            self.text = pattern.sub(value, self.text)
        # print(self.text)

    def replace_emsho(self):
        emso_regex = r"\b[0123]\d(0[1-9]|1[0-2])9\d{2}50[05]\d{3}\b"
        vat_regex = r"\b(SI)?\s?\d{8}\b"
        for info in re.finditer(emso_regex, self.text, re.M):
            pp = info.group(0)
            emso_factor_map = [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
            control_digit = sum([int(pp[i]) * emso_factor_map[i] for i in range(12)]) % 11
            control_digit = 0 if control_digit == 0 else 11 - control_digit
            is_emso = control_digit == int(pp[12])
            if pp not in self.emso_mapper and is_emso:
                new_val = rstr.xeger(emso_regex)
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.emso_mapper[pp] = new_val
            for key, val in self.emso_mapper.items():
                self.text = self.text.replace(key, val)

        for vat in re.finditer(vat_regex, self.text, re.MULTILINE):
            pp = vat.group(0)
            if pp not in self.vat_mapper:
                calculated = self._calc_new_val("DANOCNA_STEVILKA", self.vat_mapper)
                new_val =  calculated if calculated != "" else self.faker.vat_id()
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.vat_mapper[pp] = new_val
            for key, val in self.vat_mapper.items():
                self.text = self.text.replace(key, val)

    def _replace_info(self, pattern, mapper, placeholder):
        for info in re.finditer(pattern, self.text, re.MULTILINE):
            pp = info.group(0)
            if pp not in mapper:
                calc = self._calc_new_val(placeholder, mapper)
                new_val = rstr.xeger(pattern) if calc == "" else calc
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                mapper[pp] = new_val
            for key, val in mapper.items():
                self.text = self.text.replace(key, val)


    def replace_bank_info(self):
        swift_regex = r"\b(?<!\[)[A-Z]{6}[A-Z2-9][A-NP-Z0-9]([A-Z0-9]{3})?\b"
        iban_regex = r"\bSI56\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b"
        cc_number_reg = r"\b[456][1-9]{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"
        self._replace_info(swift_regex, self.bank_info_mapper["swift"], "SWIFT" if self.mode != "low" else "BANK_INFO")
        self._replace_info(iban_regex, self.bank_info_mapper["iban"], "IBAN" if self.mode != "low" else "BANK_INFO")
        for info in re.finditer(cc_number_reg, self.text, re.M):
            pp = info.group(0)
            if pp not in self.bank_info_mapper["cc_number"]:
                new_val = rstr.xeger(cc_number_reg).replace("-", "").replace(" ", "")
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.bank_info_mapper["cc_number"][pp] = new_val
            for key, val in self.bank_info_mapper["cc_number"].items():
                self.text = self.text.replace(key, val)

    def replace_organizations(self):
        organizations = self.ner_predictor.tag_dist["org"] if "org" in self.ner_predictor.tag_dist else []
        for organization in organizations:
            if organization not in self.org_mapper.keys():
                new_val = self.faker.company()
                comp = self._calc_new_val("ORGANIZATION", self.org_mapper)
                self.org_mapper[organization] = new_val if comp == "" else comp
        for key, val in self.org_mapper.items():
            pattern = re.compile(re.escape(key), re.I)
            self.text = pattern.sub(val, self.text)

    def replace_misc(self):
        misc = []
        if "misc" in self.ner_predictor.tag_dist.keys():
            misc.extend(self.ner_predictor.tag_dist["misc"])
        if "pro" in self.ner_predictor.tag_dist.keys():
            misc.extend(self.ner_predictor.tag_dist["pro"])

        for mis in misc:
            if mis not in self.misc_mapper.keys():
                parts = mis.split(" ")
                placeholder = "MISC"
                if len(parts) > 0 and parts[-1] in self.pos_predictor.word_to_tag:
                    pos_tag = self.pos_predictor.word_to_tag[parts[-1]]
                    if pos_tag:
                        if pos_tag[0] == "N":
                            # placeholder = "NOUN"
                            if pos_tag[1] == "c":
                                placeholder = "COMMON_NOUN"
                            else:
                                placeholder = "PROPER_NOUN"
                            if pos_tag[3] == "s":
                                placeholder = f"SINGULAR_{placeholder}"
                            elif pos_tag[3] == "d":
                                placeholder = f"DUAL_{placeholder}"
                            else:
                                placeholder = f"PLURAL_{placeholder}"
                        if pos_tag[0] == "V":
                            placeholder = "VERB"
                        if pos_tag[0] == "A":
                            placeholder = "ADJECTIVE"
                        if pos_tag[0] == "P":
                            placeholder = "PRONOUN"
                calcu = self._calc_new_val(placeholder, self.misc_mapper)
                new_val = self.faker.word() if calcu == "" else calcu
                if mis[-1] in string.punctuation:
                    new_val = f"{new_val}{mis[-1]}"
                self.misc_mapper[mis] = new_val

    def find_email_from_person(self, person: str):
        emails = self.mail_mapper.keys()
        lower_person = person.lower()
        name_surname = person.split(" ")
        potential_usernames = []
        if len(name_surname) > 1:
            potential_usernames.extend([
                person.replace(" ", "_"),
                person.replace(" ", ""),
                person.replace(" ", "."),
                lower_person.replace(" ", "_"),
                lower_person.replace(" ", ""),
                lower_person.replace(" ", "."),
                f"{name_surname[0][0]}{name_surname[1]}",
                f"{name_surname[1]}{name_surname[0][0]}",
                f"{name_surname[0]}{name_surname[1][0]}",
            ])
        else:
            potential_usernames.append(person)
        for email in emails:
            username = email.split("@")
            if len(username) > 1:
                username = re.sub(r"\d+", "", username[0])
                has_email = any([x for x in potential_usernames if nltk.edit_distance(x, username) < 3])
                if has_email:
                    return email
        return ""

    def replace_person(self):
        word_pos_ner = defaultdict(dict)
        females = defaultdict(str)
        males = defaultdict(str)
        others = defaultdict(str)
        for word, tag in self.ner_predictor.word_to_tag.items():
            if "per" in self.ner_predictor.tag_dist and word in self.ner_predictor.tag_dist["per"]:
                last_word = word.split(" ")
                new_name = ""
                if len(last_word) > 1 and last_word[0] in self.pos_predictor.word_to_tag.keys():
                    pos_tag = self.pos_predictor.word_to_tag[last_word[0]]
                    if pos_tag[0] == "n" and pos_tag[1] == "p":
                        if pos_tag[2] == "m":
                            new_name = f"{self.faker.first_name_male()} {self.faker.last_name_male()}"
                            new_m = self._calc_new_val("MALE", males)
                            new_name = new_name if new_m == "" else new_m
                            males[word] = new_name
                        elif pos_tag[2] == "f":
                            new_name = f"{self.faker.first_name_female()} {self.faker.last_name_female()}"
                            new_m = self._calc_new_val("FEMALE", females)
                            new_name = new_name if new_m == "" else new_m
                            females[word] = new_name
                        else:
                            new_name = f"{self.faker.first_name_nonbinary()} {self.faker.last_name_nonbinary()}"
                            new_m = self._calc_new_val("NON_BINARY", others)
                            new_name = new_name if new_m == "" else new_m
                            others[word] = new_name
                    else:
                        new_name = f"{self.faker.first_name_nonbinary()} {self.faker.last_name_nonbinary()}"
                        new_m = self._calc_new_val("NON_BINARY", others)
                        new_name = new_name if new_m == "" else new_m
                        others[word] = new_name
                else:
                    new_name = f"{self.faker.first_name_nonbinary()} {self.faker.last_name_nonbinary()}"
                    new_m = self._calc_new_val("NON_BINARY", others)
                    new_name = new_name if new_m == "" else new_m
                    others[word] = new_name
                    # new_name = f"{self.faker.first_name_male()} {self.faker.last_name_male()}" if  pos_tag[2] == "m" else f"{self.faker.first_name_female()} {self.faker.last_name_female()}"
                if word[-1] in string.punctuation:
                    new_name += word[-1]
                name_pattern = re.compile(re.escape(word), re.I)
                self.text = name_pattern.sub(new_name, self.text)
                potential_email = self.find_email_from_person(word)
                if potential_email != "":
                    self.mail_mapper[
                        potential_email] = f"{new_name.replace(' ', '.').lower()}@{self.faker.free_email_domain()}"
                    self.text = self.text.replace(potential_email, self.mail_mapper[potential_email])
        # print(self.text)

    def _find_dates(self):
        basic_reg = r"\b[012]\d[\./-][01]\d[\./-][12]\d{3}|[1][12][\./-][12]\d{3}|[0]\d[\./-][12]\d{3}|\d[\./-][12]\d{3}\b"  # 20.04.2020 or 20/04/2022 or 02.2022
        text_reg = r"\b(\d{1,2}\.\s\w+\b\s?(?:[12]\d{3})?)|((leta|letu)\s\d{4})\b"
        for date in re.finditer(basic_reg, self.text, re.I | re.M):
            date = date.group(0)
            if date not in self.date_mapper:
                date_obj = dateparser.parse(date)
                date_start = date_obj - datetime.timedelta(weeks=20)
                date_format = "%d.%m.%Y"
                if "/" in date:
                    date_format = "%d/%m/%Y"
                if "-" in date:
                    date_format = "%d-%m-%Y"
                calc = self._calc_new_val("DATE", self.date_mapper)
                new_val = self.faker.date_between(start_date=date_start, end_date="today").strftime(
                    date_format) if calc == "" else calc
                if date[-1] in string.punctuation:
                    new_val += date[-1]
                self.date_mapper[date] = new_val

        for da in re.finditer(text_reg, self.text, re.I | re.M):
            parts = da.group(0).strip().split(" ")
            calcu = self._calc_new_val("DATE", self.date_mapper)
            new_val = ""
            print(parts)
            if parts[0] == "leta":
                year = int(parts[1]) if parts[1].isdecimal() else 2000
                year = max([random.randint(year - 10, year + 10), datetime.date.today().year])
                new_val = f"leta {year}"
                # new_val = f"leta {random.randint(1990, 2022)}"
            else:
                month = parts[1].strip()
                month = self.classla(month).get("lemma")[0]
                if month in MONTHS:
                    month_no = MONTHS.index(month) + 1
                    total_days = monthrange(datetime.datetime.now().year, month_no)[1]
                    year = ""
                    if len(parts) > 2:
                        year = int(parts[2])
                        year = random.randint(year - 10, year + 10)
                    new_val = f"{random.randint(1, total_days)}. {parts[1]} {year}" if calcu == "" else calcu
            vval = da.group(0)
            if new_val != "":
                if vval[-1] in string.punctuation:
                    new_val += vval[-1]
                self.date_mapper[da.group(0)] = new_val

        for key, val in self.date_mapper.items():
            self.text = self.text.replace(key, val)

    def replace_dates(self):
        # dates = search_dates(self.text, settings={'DATE_ORDER': 'DMY'})
        self._find_dates()

    def replace_licence_plate(self):
        plate_reg = r"\b(CE|GO|KK|KP|KR|LJ|MB|MS|NM|PO|SG)\s?(([0-9]{3}\-?[A-Z]{2})|([A-Z]{2}[\s-]?\d{3})|([0-9]{2}[\s-]?[A-Z]{3}))\b"
        self._replace_info(plate_reg, self.plate_mapper, "LICENCE_PLATE")


    def replace_events(self):
        for word, tag in self.ner_predictor.word_to_tag.items():
            if word in self.ner_predictor.tag_dist["evt"]:
                if word not in self.event_mapper:
                    calc = self._calc_new_val("EVENT", self.event_mapper)
                    new_val = calc if calc != "" else random.choice(EVENT_NAMES)
                    if word[-1] in string.punctuation:
                        new_val += word[-1]
                    self.event_mapper[word] = new_val

        for key, value in self.event_mapper.items():
            pat = re.compile(re.escape(key), re.I)
            self.text = pat.sub(value, self.text)


    def replace_personal_info(self):
        passport_pattern = r"\b[P][A-Z0-9][0-9]{7}\b"
        pid_pattern = r"\b(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[012])([9][2-9][0-9]|0[012][0-9])[0-9]{6}\b"
        new_pid_pattern = r"\b[I][A-Z][0-9]{7}\b"
        for passport in re.finditer(passport_pattern, self.text, re.I):
            if passport not in self.personal_info_mapper["passport_no"]:
                letters = "".join(random.choices(string.ascii_uppercase, k=2))
                numbers = random.randint(1000000, 9999999)
                self.personal_info_mapper["passport_no"][passport] = f"{letters}{numbers}"
        for pid in re.finditer(pid_pattern, self.text, re.I):
            if pid not in self.personal_info_mapper["pid"]:
                self.personal_info_mapper["pid"][pid] = random.randint(100000000, 999999999)
        for pid in re.finditer(new_pid_pattern, self.text, re.I):
            if pid not in self.personal_info_mapper["pid"]:
                letters = "".join(random.choices(string.ascii_uppercase, k=2))
                numbers = random.randint(1000000, 9999999)
                self.personal_info_mapper["pid"][pid] = f"{letters}{numbers}"
        for key, val in self.personal_info_mapper["passport_no"]:
            self.text = self.text.replace(key, val)
        for key, val in self.personal_info_mapper["pid"]:
            self.text = self.text.replace(key, val)

    def reset_mappers(self):
        self.phone_mapper: defaultdict = defaultdict(str)
        self.name_mapper: defaultdict = defaultdict(str)
        self.mail_mapper: defaultdict = defaultdict(str)
        self.org_mapper: defaultdict = defaultdict(str)
        self.address_mapper: defaultdict = defaultdict(str)
        self.emso_mapper: defaultdict = defaultdict(str)
        self.vat_mapper: defaultdict = defaultdict(str)
        self.misc_mapper: defaultdict = defaultdict(str)
        self.person_mapper: defaultdict = defaultdict(str)
        self.date_mapper: defaultdict = defaultdict(str)
        self.plate_mapper: defaultdict = defaultdict(str)
        self.event_mapper: defaultdict = defaultdict(str)
        self.bank_info_mapper: dict = {"swift": defaultdict(str), "iban": defaultdict(str),
                                       "cc_number": defaultdict(str)}
        self.personal_info_mapper: dict = {"passport_no": defaultdict(str), "pid": defaultdict(str)}
        self.pos_predictor.reset_params()
        self.ner_predictor.reset_params()



#
# if __name__ == "__main__":
#     # nltk.download('omw-1.4')
#     pipeline = Pipeline("Slovenske prvake že v nedeljo, 23. septembra, v velenjski Rdeči dvorani čaka večni derbi z Gorenjem v okviru tretjega kroga lige NLB, v četrtek, 29. septembra, pa še gostovanje pri španski Barceloni. Katalonski velikan, kjer igrata tudi slovenska reprezentanta Blaž Janc in Domen Makuc, je v zadnjih dveh sezonah slavil v Ligi Prvakov.")
#     pipeline.start_predictions()
#     print(pipeline.pos_predictor.tag_dist)
#     print(pipeline.ner_predictor.tag_dist)
#     pipeline.replace_organizations()
#     pipeline.replace_dates()
#     pipeline.replace_misc()
#     pipeline.replace_address()
#     pipeline.replace_mail()
#     pipeline.replace_phones()
#     pipeline.replace_emsho()
#     pipeline.replace_bank_info()
#     pipeline.replace_person()
#     pipeline.replace_events()
#     print(pipeline.text)