import datetime
import logging
import os
import sys
from calendar import monthrange
from collections import defaultdict
import re
import random
from faker import Faker

from vp_ta_constants import COUNTRIES, SLOVENIAN_MUNICIPALITIES, SLOVENIAN_CITIES, MONTHS, EVENT_NAMES
import classla
import rstr
import nltk
# from dateparser.search import search_dates
import dateparser
import string
import requests

class Pipeline:
    def __init__(self, text):
        nltk.download('omw-1.4')
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
        self.classla = classla.Pipeline('sl', processors='tokenize,pos,lemma')
        Faker.seed(64)
        self.mode = "readable"
        self.faker: Faker = Faker("sl_SI")
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrainL1OStrategy')

        self.ner_tag_dist: defaultdict = defaultdict(list)
        self.ner_word_to_tag = defaultdict(str)

        self.pos_tag_dist: defaultdict = defaultdict(list)
        self.pos_word_to_tag = defaultdict(str)
        # classla.download('sl', processors=processors)

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
                calculated = self._calc_new_val("PHONE", self.phone_mapper)
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
        free_domains = ["gmail.com", "yahoo.com", "live.com", "hotmail.com", "otulook.com"]
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
        locations = self.ner_tag_dist["loc"] if "loc" in self.ner_tag_dist else []
        print(locations)
        for location in locations:
            placeholder = "ADDRESS"
            lower_loc = self.classla(location.lower()).get("lemma")[0]
            if location not in self.address_mapper and lower_loc not in COUNTRIES and lower_loc not in SLOVENIAN_MUNICIPALITIES and lower_loc not in SLOVENIAN_CITIES:
                placeholder = "STREET_ADDRESS"
                self.address_mapper[location] = self.faker.street_name()
            elif location not in self.address_mapper and lower_loc in SLOVENIAN_MUNICIPALITIES:
                self.address_mapper[location] = self.faker.administrative_unit()
                placeholder = "ADMINISTRATIVE_UNIT"
            elif location not in self.address_mapper and lower_loc in SLOVENIAN_CITIES:
                self.address_mapper[location] = self.faker.city()
                placeholder = "CITY"
            else:
                self.address_mapper[location] = self.faker.country()
                placeholder = "LOCATION"
            calculated = self._calc_new_val("LOCATION" if self.mode is not "low" else placeholder, self.address_mapper)
            new_val = calculated if calculated != "" else self.address_mapper[location]
            if location[-1] in string.punctuation:
                new_val = f"{new_val}{location[-1]}"
            self.address_mapper[location] = new_val

        for key, value in self.address_mapper.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            self.text = pattern.sub(value, self.text)
        # print(self.text)

    def replace_emsho(self):
        emso_regex = r"\b[0123]\d(0[1-9]|1[0-2])9\d{2}50\d{4}\b"
        vat_regex = r"\b(SI)?\s?\d{8}\b"
        for info in re.finditer(emso_regex, self.text, re.M):
            pp = info.group(0)
            emso_factor_map = [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
            control_digit = sum([int(pp[i]) * emso_factor_map[i] for i in range(12)]) % 11
            control_digit = 0 if control_digit == 0 else 11 - control_digit
            is_emso = control_digit == int(pp[12])
            if pp not in self.emso_mapper and is_emso:
                calc = self._calc_new_val("EMŠO", self.emso_mapper)
                new_val = rstr.xeger(emso_regex) if calc == "" else calc
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.emso_mapper[pp] = new_val
            for key, val in self.emso_mapper.items():
                self.text = self.text.replace(key, val)

        for vat in re.finditer(vat_regex, self.text, re.MULTILINE):
            pp = vat.group(0)
            if pp not in self.vat_mapper:
                calculated = self._calc_new_val("VAT_ID", self.vat_mapper)
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
        swift_regex = r"\b[A-Z]{6}[A-Z2-9][A-NP-Z0-9]([A-Z0-9]{3})?\b"
        iban_regex = r"\bSI56\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b"
        cc_number_reg = r"\b[3456][1-9]{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"

        self._replace_info(swift_regex, self.bank_info_mapper["swift"], "SWIFT" if self.mode is not "low" else "BANK_INFO")
        self._replace_info(iban_regex, self.bank_info_mapper["iban"], "IBAN" if self.mode is not "low" else "BANK_INFO")
        for info in re.finditer(cc_number_reg, self.text, re.M):
            pp = info.group(0)
            if pp not in self.bank_info_mapper["cc_number"]:
                calc = self._calc_new_val("CREDIT_CARD_NUMBER" if self.mode is not "low" else "BANK_INFO", self.bank_info_mapper["cc_number"])
                new_val = self.faker.credit_card_number() if calc == "" else calc
                if pp[-1] in string.punctuation:
                    new_val = f"{new_val}{pp[-1]}"
                self.bank_info_mapper["cc_number"][pp] = new_val
            for key, val in self.bank_info_mapper["cc_number"].items():
                self.text = self.text.replace(key, val)

    def replace_organizations(self):
        organizations = self.ner_tag_dist["org"] if "org" in self.ner_tag_dist else []
        for organization in organizations:
            if organization not in self.org_mapper.keys():
                comp = self._calc_new_val("ORG", self.org_mapper)
                new_val = self.faker.company() if comp == "" else comp
                if organization[-1] in string.punctuation:
                    new_val = f"{new_val}{organization[-1]}"
                self.org_mapper[organization] = new_val
        for key, val in self.org_mapper.items():
            pattern = re.compile(re.escape(key), re.I)
            self.text = pattern.sub(val, self.text)

    def replace_misc(self):
        misc = []
        if "misc" in self.ner_tag_dist.keys():
            misc.extend(self.ner_tag_dist["misc"])
        if "pro" in self.ner_tag_dist.keys():
            misc.extend(self.ner_tag_dist["pro"])

        for mis in misc:
            if mis not in self.misc_mapper.keys():
                parts = mis.split(" ")
                pos_tag = self.pos_word_to_tag[parts[-1]]
                placeholder = "MISC"
                if pos_tag:
                    if pos_tag[0] == "N":
                        placeholder = "NOUN"
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

        for key, val in self.misc_mapper.items():
            pattern = re.compile(re.escape(key), re.I)
            self.text = pattern.sub(val, self.text)

    def find_email_from_person(self, person: str):
        emails = self.mail_mapper.keys()
        if self.mode == "readable":
            lower_person = person.lower()
            name_surname = lower_person.split(" ")
            potential_usernames = []
            if len(name_surname) > 1:
                potential_usernames.extend([
                    person.replace(" ", "_"),
                    person.replace(" ", ""),
                    person.replace(" ", "."),
                    "".join(reversed(name_surname)),
                    ".".join(reversed(name_surname)),
                    "_".join(reversed(name_surname)),
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

    def replace_events(self):
        for word, tag in self.ner_word_to_tag.items():
            if word in self.ner_tag_dist["evt"]:
                if word not in self.event_mapper:
                    calc = self._calc_new_val("EVENT", self.event_mapper)
                    new_val = calc if calc != "" else random.choice(EVENT_NAMES)
                    if word[-1] in string.punctuation:
                        new_val += word[-1]
                    self.event_mapper[word] = new_val

        for key, value in self.event_mapper.items():
            pat = re.compile(re.escape(key), re.I)
            self.text = pat.sub(value, self.text)



    def replace_person(self):
        word_pos_ner = defaultdict(dict)
        females = defaultdict(str)
        males = defaultdict(str)
        others = defaultdict(str)
        for word, tag in self.ner_word_to_tag.items():
            if word in self.ner_tag_dist["per"]:
                last_word = word.split(" ")
                new_name = ""
                if len(last_word) > 1 and last_word[0] in self.pos_word_to_tag.keys():
                    pos_tag = self.pos_word_to_tag[last_word[0]]
                    if pos_tag[0] == "n" and pos_tag[1] == "p":
                        if pos_tag[2] == "m":
                            new_name = f"{self.faker.first_name_male()} {self.faker.last_name_male()}"
                            new_m = self._calc_new_val("MALE", males)
                            new_name =  new_name if new_m == "" else new_m
                            males[word] = new_name
                        elif pos_tag[2] == "f":
                            new_name = f"{self.faker.first_name_female()} {self.faker.last_name_female()}"
                            new_m = self._calc_new_val("FEMALE", males)
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
                    self.mail_mapper[potential_email] = f"{new_name.replace(' ', '.').lower()}@{self.faker.free_email_domain()}"
                    self.text = self.text.replace(potential_email, self.mail_mapper[potential_email])

    def _find_dates(self):
        basic_reg = r"\b[012]\d[\./-][01]\d[\./-][12]\d{3}|[1][12][\./-][12]\d{3}|[0]\d[\./-][12]\d{3}|\d[\./-][12]\d{3}\b" # 20.04.2020 or 20/04/2022 or 02.2022
        text_reg = r"\b(\d{1,2}\.\s\w+\b\s?(?:[12]\d{3})?)|(leta\s\d{4})\b"
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
                new_val = self.faker.date_between(start_date=date_start, end_date="today").strftime(date_format) if calc == "" else calc
                if date[-1] in string.punctuation:
                    new_val += date[-1]
                self.date_mapper[date] = new_val

        for da in re.finditer(text_reg, self.text, re.I | re.M):
            parts = da.group(0).strip().split(" ")
            calcu = self._calc_new_val("DATE", self.date_mapper)
            new_val = ""
            if parts[0] == "leta":
                new_val = f"leta {random.randint(1990, 2022)}"
            else:
                month = parts[1].strip()
                month = self.classla(month).get("lemma")[0]
                if month in MONTHS:
                    new_month = ""

                    month_no = MONTHS.index(month) + 1
                    total_days = monthrange(datetime.datetime.now().year, month_no)[1]
                    year = ""
                    if len(parts) > 2:
                        year = int(parts[2])
                        year = max([random.randint(year - 10, year + 10), datetime.date.year])
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

    def replace_personal_info(self):
        passport_pattern = r"\b[P][A-Z0-9][0-9]{7}\b"
        pid_pattern = r"\b(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[012])([9][2-9][0-9]|0[012][0-9])[0-9]{6}\b"
        new_pid_pattern = r"\b[I][A-Z][0-9]{7}\b"
        for passport in re.finditer(passport_pattern, self.text, re.I):
            if passport not in self.personal_info_mapper["passport_no"]:
                letters = "".join(random.choices(string.ascii_uppercase, k=2))
                numbers = random.randint(1000000, 9999999)
                calcu = self._calc_new_val("PASSPORT", self.personal_info_mapper["passport_no"])
                new_val = calcu if calcu != "" else f"{letters}{numbers}"
                if passport[-1] in string.punctuation:
                    new_val += passport[-1]
                self.personal_info_mapper["passport_no"][passport] = new_val

        for pid in re.finditer(pid_pattern, self.text, re.I):
            if pid not in self.personal_info_mapper["pid"]:
                calcu = self._calc_new_val("PID", self.personal_info_mapper["pid"])
                new_val = calcu if calcu != "" else random.randint(100000000, 999999999)
                if pid[-1] in string.punctuation:
                    new_val += pid[-1]
                self.personal_info_mapper["pid"][pid] = new_val

        for pid in re.finditer(new_pid_pattern, self.text, re.I):
            if pid not in self.personal_info_mapper["pid"]:
                letters = "".join(random.choices(string.ascii_uppercase, k=2))
                numbers = random.randint(1000000, 9999999)
                calcu = self._calc_new_val("PID", self.personal_info_mapper["pid"])
                new_val = calcu if calcu != "" else f"{letters}{numbers}"
                if pid[-1] in string.punctuation:
                    new_val += pid[-1]
                self.personal_info_mapper["pid"][pid] = new_val

        for key, val in self.personal_info_mapper["passport_no"]:
            self.text = self.text.replace(key, val)
        for key, val in self.personal_info_mapper["pid"]:
            self.text = self.text.replace(key, val)

    def fetch_ner_data(self):
        ip = os.getenv("SLO_NER_SERVICE_HOST", "34.116.147.57")
        ser_port = os.getenv("SLO_NER_SERVICE_PORT", "5000")
        self.logger.info(ser_port)
        self.logger.info(ip)
        self.logger.info(os.getenv("SLO_NER_PORT", "5000"))
        url = f"http://{ip}:{ser_port}/ner/predict"
        data = {'text': self.text}
        x = requests.post(url, json=data).json()
        self.logger.info(x)
        self.ner_tag_dist = x["tag_dist"]
        self.ner_word_to_tag = x["word_to_tag"]

    def fetch_pos_data(self):
        ip = os.getenv("SLO_POS_SERVICE_HOST", "34.116.166.137")
        port = os.getenv("SLO_POS_SERVICE_PORT", "5030")
        url = f"http://{ip}:{port}/pos/predict"
        # url = "http://pos:5030/pos/predict"
        data = {'text': self.text}
        x = requests.post(url, json=data).json()
        # self.logger.info(x)
        self.pos_tag_dist = x["tag_dist"]
        self.pos_word_to_tag = x["word_to_tag"]

    def fetch_coref_text(self):
        ip = os.getenv("SLO_COREF_SERVICE_HOST", "10.104.1.66")
        port = os.getenv("SLO_COREF_SERVICE_PORT", "5020")
        url = f"http://{ip}:{port}/coref/resolve"
        # url = "http://coref:5020/coref/resolve"
        data = {'text': self.text}
        x = requests.post(url, json=data).json()
        self.logger.info("response", x)
        self.logger.info(x)
        self.text = x["text"]

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

    def start_predictions(self, text, mode="readable"):

        self.reset_mappers()

        self.text = text
        self.mode = mode

        self.fetch_coref_text()
        self.fetch_ner_data()
        self.fetch_pos_data()

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