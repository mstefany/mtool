import argparse
import re
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from peewee import (
    DoesNotExist,
    Model,
    CharField,
    ForeignKeyField,
    IntegerField,
    TextField,
    TimestampField,
)
from playhouse.sqlite_ext import SqliteExtDatabase, FTSModel, FTS5Model, SearchField
from typing import List


DB_FILE_PATH = "storage.db"

TELEPHONE_NUMBER_REGEX = r"^\+?\d{9,}$"

SMS_PROTOCOL = {0: "SMS"}

SMS_TYPE = {
    1: "Received",
    2: "Sent",
    3: "Draft",
    4: "Outbox",
    5: "Failed",
    6: "Queued",
}

SMS_READ = {0: "Unread", 1: "Read"}

SMS_STATUS = {-1: "None", 0: "Complete", 32: "Pending", 64: "Failed"}


# Database initialization
sqlite_db = SqliteExtDatabase(
    DB_FILE_PATH, pragmas={"journal_mode": "wal", "foreign_keys": 1}
)


# Step 1: Define the SMSMessage data class to structure messages
@dataclass(frozen=True)
class Message:
    # <sms protocol="0" address="608" date="1364118303239" type="1" subject="null" body="[REDACTED]" toa="null" sc_toa="null" service_center="+420608005684" read="1" status="-1" locked="0" epoch_sent="1364121662000" sub_id="2" readable_date="Mar 24, 2013 10:45:03" contact_name="(Unknown)" />
    protocol: int
    address: str
    epoch: int
    type: int
    subject: str
    body: str
    service_center: str
    read: int
    status: int
    locked: int
    epoch_sent: int
    sub_id: int
    readable_date: str
    contact_name: str


@dataclass(frozen=True)
class Protocol:
    protocol: int
    description: str


@dataclass(frozen=True)
class SmsType:
    type: int
    description: str


@dataclass(frozen=True)
class SmsReadStatus:
    read: int
    description: str


@dataclass(frozen=True)
class SmsStatus:
    type: int
    description: str


@dataclass(frozen=True)
class Contact:
    address: str
    last: str
    first: str = ""
    middle: str = ""
    prefix: str = ""
    suffix: str = ""


# Step 2: Define the Peewee Model for the FTS table
class BaseModel(Model):
    class Meta:
        database = sqlite_db


class Protocols(BaseModel):
    protocol = IntegerField(unique=True, primary_key=True)
    description = TextField(null=True)


class SmsTypes(BaseModel):
    type = IntegerField(unique=True, primary_key=True)
    description = TextField(null=True)


class SmsReadStatuses(BaseModel):
    read = IntegerField(unique=True, primary_key=True)
    description = TextField(null=True)


class SmsStatuses(BaseModel):
    status = IntegerField(unique=True, primary_key=True)
    description = TextField(null=True)


class Contacts(BaseModel):
    prefix = CharField(null=True)
    first = CharField(null=True)
    middle = CharField(null=True)
    last = CharField()
    suffix = CharField(null=True)
    nickname = CharField(null=True)

    class Meta:
        indexes = (
            # create a unique on from/to/date
            (("first", "last"), True),
        )


class TelephoneNumbers(BaseModel):
    number = CharField(unique=True)
    owner = ForeignKeyField(Contacts, null=True, backref="numbers")
    description = TextField(null=True)


class EmailAddresses(BaseModel):
    email = CharField(unique=True)
    owner = ForeignKeyField(Contacts, null=True, backref="emailaddresses")
    description = TextField(null=True)


class Smses(BaseModel):
    protocol = IntegerField()
    address_original = CharField()
    address = ForeignKeyField(TelephoneNumbers, null=True, backref="smses")
    epoch = TimestampField(resolution=3)
    type = IntegerField()
    subject = TextField(null=True)
    body = TextField()
    service_center = CharField()
    read = IntegerField()
    status = IntegerField()
    locked = IntegerField()
    epoch_sent = TimestampField(resolution=3)
    sub_id = IntegerField()


class SmsesIndexModel(FTSModel):
    body = SearchField()

    class Meta:
        database = sqlite_db
        options = {"content": Smses.body}  # <-- specify data source.


# Step 3: Set up database and FTS table
class Database:
    def __init__(self, db: SqliteExtDatabase):
        self.db = db

    def setup(self):
        """Create the SQLite database and FTS table using Peewee."""
        self.db.connect()
        self.db.create_tables(
            [
                Protocols,
                SmsTypes,
                SmsReadStatuses,
                SmsStatuses,
                Contacts,
                TelephoneNumbers,
                EmailAddresses,
                Smses,
                SmsesIndexModel,
            ],
            safe=True,
        )
        self.db.close()

    def setup_known_metadata(self) -> None:
        """Initialize some tables with known properties from SMS Backup & Restore"""

        self.db.connect()

        with self.db.atomic():
            for p in get_protocols():
                if not Protocols.get_or_none(protocol=p.protocol):
                    Protocols.create(protocol=p.protocol, description=p.description)

            for t in get_sms_types():
                if not SmsTypes.get_or_none(type=t.type):
                    SmsTypes.create(type=t.type, description=t.description)

            for r in get_sms_read_statuses():
                if not SmsReadStatuses.get_or_none(read=r.read):
                    SmsReadStatuses.create(read=r.read, description=r.description)

            for s in get_sms_statuses():
                if not SmsStatuses.get_or_none(status=s.status):
                    SmsStatuses.create(status=s.status, description=s.description)

        self.db.close()

    def save(self, messages: List[Message], contacts: List[Contact]):
        """Save parsed SMS messages into the SQLite database."""
        self.db.connect()

        for cnt in contacts:
            with self.db.atomic():
                # if not TelephoneNumbers.get_or_none(number=cnt.address):
                if not TelephoneNumbers.get_or_none(number=cnt.address):
                    # there's no telephone number, but there might be contact already
                    contact, _ = Contacts.get_or_create(first=cnt.first, last=cnt.last)

                    if cnt.prefix != "":
                        contact.prefix = cnt.prefix

                    if cnt.middle != "":
                        contact.middle = cnt.middle

                    if cnt.suffix != "":
                        contact.suffix = cnt.suffix

                    contact.save()

                    TelephoneNumbers.create(number=cnt.address, owner=contact.id)

        # Insert each message into the FTS table
        with self.db.atomic():  # Wrap insertions in a transaction for performance
            for msg in messages:
                tn = TelephoneNumbers.select().where(
                    TelephoneNumbers.number.contains(msg.address)
                )
                Smses.create(
                    protocol=msg.protocol,
                    address_original=msg.address,
                    address=tn,
                    epoch=msg.epoch,
                    type=msg.type,
                    subject=msg.subject,
                    body=msg.body,
                    service_center=msg.service_center,
                    read=msg.read,
                    status=msg.status,
                    locked=msg.locked,
                    epoch_sent=msg.epoch_sent,
                    sub_id=msg.sub_id,
                )

        self.db.close()


def get_protocols() -> List[Protocols]:
    protocols = []

    for key in SMS_PROTOCOL.keys():
        protocols.append(Protocol(protocol=key, description=SMS_PROTOCOL[key]))

    return protocols


def get_sms_types() -> List[SmsTypes]:
    types = []

    for key in SMS_TYPE.keys():
        types.append(SmsTypes(type=key, description=SMS_TYPE[key]))

    return types


def get_sms_read_statuses() -> List[SmsReadStatuses]:
    statuses = []

    for key in SMS_READ.keys():
        statuses.append(SmsReadStatuses(read=key, description=SMS_READ[key]))

    return statuses


def get_sms_statuses() -> List[SmsStatuses]:
    statuses = []

    for key in SMS_STATUS.keys():
        statuses.append(SmsStatuses(status=key, description=SMS_STATUS[key]))

    return statuses


# Step 4: Parse the SMS Backup & Restore XML file
def parse_sms_backup_and_restore_xml(file_path: str) -> List[Message]:
    """Parse the XML file and extract SMS messages."""
    messages = []

    # Parse the XML file using ElementTree
    tree = ET.parse(file_path)
    root = tree.getroot()

    for sms in root.findall("sms"):
        # <sms protocol="0" address="608" date="1364118303239" type="1" subject="null" body="[REDACTED]" toa="null" sc_toa="null" service_center="+420608005684" read="1" status="-1" locked="0" epoch_sent="1364121662000" sub_id="2" readable_date="Mar 24, 2013 10:45:03" contact_name="(Unknown)" />

        protocol = int(sms.get("protocol"))
        address = sms.get("address")
        epoch = int(sms.get("date"))
        type = int(sms.get("type"))

        if sms.get("subject") != "null":
            subject = sms.get("subject")
        else:
            subject = ""

        body = sms.get("body")

        if sms.get("service_center") != "null":
            service_center = sms.get("service_center")
        else:
            service_center = ""

        read = int(sms.get("read"))
        status = int(sms.get("status"))
        locked = int(sms.get("locked"))
        epoch_sent = int(sms.get("date_sent"))
        sub_id = int(sms.get("sub_id"))
        readable_date = sms.get("readable_date")
        contact_name = sms.get("contact_name")

        messages.append(
            Message(
                protocol=protocol,
                address=address,
                epoch=epoch,
                type=type,
                subject=subject,
                body=body,
                service_center=service_center,
                read=read,
                status=status,
                locked=locked,
                epoch_sent=epoch_sent,
                sub_id=sub_id,
                readable_date=readable_date,
                contact_name=contact_name,
            )
        )

    return messages


def extract_contacts(messages: List[Message]) -> List[Contact]:
    """Try to extract contact information from messages"""
    contacts = []
    known_numbers = set()

    for msg in messages:
        if not is_valid_phone_number(msg.address):
            continue

        if "Unknown" in msg.contact_name:
            continue

        if msg.address in known_numbers:
            continue
        else:
            known_numbers.add(msg.address)

        contacts.append(guess_contact(msg.address, msg.contact_name))

    return contacts


def is_valid_phone_number(number: str) -> bool:
    return bool(re.match(TELEPHONE_NUMBER_REGEX, number))


def guess_contact(address: str, contact_name: str) -> Contact:
    prefix = ""
    suffix = ""
    words = contact_name.split()

    if len(words) > 1 and "." in words[0]:
        # assume academic title
        prefix = words[0]
        words = words[1:]

    if len(words) > 1 and "." in words[-1]:
        # assume Sr., Jr., etc.
        suffix = words[-1]
        words = words[:-1]

    if len(words) > 1 and words[-1] == str(words[-1]).upper():
        # assume EFA, MFA, etc.
        suffix = words[-1]
        words = words[:-1]

    # hard to tell if name or surname or something, assume surname-only
    if len(words) == 1:
        return Contact(address=address, prefix=prefix, last=words[0], suffix=suffix)
    elif len(words) == 2:  # assume standard first + last
        last = str(words[1]).replace(",", "")
        return Contact(
            address=address, prefix=prefix, first=words[0], last=last, suffix=suffix
        )
    else:
        last = str(words[-1]).replace(",", "")
        return Contact(
            address=address,
            prefix=prefix,
            first=words[0],
            middle=" ".join(words[1:-1]),
            last=last,
            suffix=suffix,
        )


def optimizer() -> None:
    SmsesIndexModel.rebuild()
    SmsesIndexModel.optimize()


# Step 6: CLI handling using argparse
def main():
    parser = argparse.ArgumentParser(
        description="SMS Backup & Restore XML parser and importer."
    )
    parser.add_argument(
        "-x",
        "--import",
        dest="xml_file",
        type=str,
        help="Path to SMS Backup & Restore XML file to import.",
    )

    parser.add_argument(
        "-i",
        "--init",
        dest="initialize",
        action="store_true",
        help="Initialize database storage",
    )

    parser.add_argument(
        "-r",
        "--rebuild",
        action="store_true",
        help="Rebuild & optimize database storage",
    )

    parsed_args = parser.parse_args()

    db = Database(sqlite_db)

    if parsed_args.initialize:
        # Step 7: Set up the database and create tables
        db.setup()
        db.setup_known_metadata()
        print("Successfully setup the database.")

    elif parsed_args.rebuild:
        optimizer()
        print(f"Successfully optimized the database.")

    elif parsed_args.xml_file:
        # Step 8: Parse the XML file
        messages = parse_sms_backup_and_restore_xml(parsed_args.xml_file)
        contacts = extract_contacts(messages)

        # Step 9: Save parsed messages into the SQLite database
        db.save(messages, contacts)

        print(f"Successfully imported {len(messages)} messages into the database.")

    else:
        parser.print_help()


# Entry point for the script
if __name__ == "__main__":
    main()
