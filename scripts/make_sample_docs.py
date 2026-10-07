"""Generate the sample documentation of the fictional Casa Aurora Boutique Hotel (5 PDFs, 15 pages).

    uv run python scripts/make_sample_docs.py

Everything is invented, including the town of Valmora, so that a language model cannot know any of it
and a correct answer proves that retrieval worked. The PDFs are written by a tiny dependency-free
generator (Helvetica text only), and the output is byte-for-byte reproducible.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

HOTEL = "Casa Aurora Boutique Hotel"
ADDRESS = "18 Harbour Lane, Valmora"

# file name -> list of (title, text): one page per entry
DOCUMENTS: dict[str, list[tuple[str, str]]] = {
    "guest_information.pdf": [
        ("Front desk and arrival",
         "The front desk is staffed every day from 8:00 to 23:00. Outside these hours, guests can ring the "
         "night bell at the main door in case of an emergency. Check-in starts at 14:00 and check-out is at "
         "12:00. A late check-out until 14:00 is available for 20 euros, subject to availability, and must be "
         "requested the day before at the front desk. Guests who expect to arrive after 23:00 must tell the "
         "hotel in advance by phone or email so that a member of staff can wait for them. Luggage can be "
         "stored free of charge on the day of arrival and on the day of departure."),
        ("Wi-Fi and connectivity",
         "Wi-Fi is free for all guests, in the rooms and in the public areas. The network is called "
         "CasaAurora_Guest and the password is printed on the key card sleeve handed over at check-in. The "
         "connection is fibre optic with a speed of 300 Mbps. A small business corner in the lobby has a "
         "printer, and guests can print 10 pages per day for free. If the connection drops, the front desk "
         "can send a technician or lend a portable router."),
        ("Parking and transport",
         "The hotel has no private car park. Guests can use the Harbour Garage, 200 metres away, at 22 euros "
         "per 24 hours; booking a space through the front desk is recommended in summer. An airport transfer "
         "costs 35 euros per car for up to three passengers and must be booked at least 24 hours in advance. "
         "The Saint Elmo tram stop is a 7-minute walk from the hotel, and the main railway station is 10 "
         "minutes away by taxi."),
        ("Laundry and extra services",
         "A laundry and dry-cleaning service is offered: items handed in before 10:00 are returned the next "
         "day by 18:00. The service is not available on Saturdays. An ironing service, or an iron with an "
         "ironing board, can be provided on request. Every room has a free safe large enough for a laptop. "
         "Umbrellas can be borrowed at the front desk when it rains."),
    ],
    "rooms_and_rates.pdf": [
        ("Rooms and rates",
         "The hotel has 31 rooms in four categories. The Standard room measures 18 square metres, overlooks "
         "the inner courtyard and has either a double bed or two twin beds; it costs from 95 euros per night "
         "in low season up to 130 euros in high season. The Superior room measures 24 square metres, has a "
         "small balcony and a partial view of the harbour, and costs between 135 and 175 euros. The Junior "
         "Suite measures 35 square metres, has a full harbour view and a soaking bathtub, and costs between "
         "190 and 250 euros. The Penthouse Suite measures 60 square metres, with a private roof terrace and "
         "an outdoor hot tub, and costs between 320 and 400 euros. Low season runs from November to March, "
         "except school holidays and the Christmas period; high season runs from June to September and "
         "during the Christmas and New Year weeks. Breakfast is included in all rates. An extra bed costs 30 "
         "euros per night and a baby cot is free on request."),
        ("Booking and cancellation policy",
         "Cancellation is free up to 72 hours before the arrival date. After that, the first night is "
         "charged. In case of a no-show on the arrival day, the whole stay is charged. Payments are accepted "
         "by Visa, Mastercard and Maestro; American Express is not accepted. A credit card imprint is taken "
         "at check-in to cover extras. For stays of four nights or more booked in high season, a deposit of "
         "20 percent is requested at booking. A non-refundable Saver rate, 10 percent cheaper than the "
         "standard rate, is also offered."),
        ("Room comfort",
         "All rooms have individually controlled air conditioning, and underfloor heating is used in winter. "
         "Windows are soundproofed and every room has blackout curtains. Quiet hours are from 22:00 to 08:00 "
         "in the corridors and the courtyard. In case of a fault, guests should contact the front desk, "
         "which can offer another room while the problem is being fixed. The air conditioning units are "
         "serviced every year in May."),
    ],
    "dining_and_wellness.pdf": [
        ("Breakfast and restaurant",
         "Breakfast is served as a buffet from 7:30 to 10:30 on weekdays and from 8:00 to 11:30 on weekends, "
         "in the courtyard. The hotel restaurant, Aurora Table, is open for lunch and dinner every day except "
         "Monday, when it is closed. It serves modern Mediterranean cuisine made with local fish and "
         "vegetables, and offers a children's menu at 12 euros. Room service is available until 21:30 with "
         "a short menu."),
        ("Rooftop pool",
         "The outdoor pool on the roof is open from May to October, from 9:00 to 20:00. It is not heated: "
         "the water is around 24 degrees in summer. Towels are provided at the pool. Children may use the "
         "pool only under the constant supervision of an adult. There is no indoor pool."),
        ("Wellness room",
         "The wellness room has a sauna and two massage rooms. It is open from 11:00 to 20:00 and access is "
         "by reservation at the front desk only. Access is free for guests staying in the Penthouse Suite; "
         "for all other guests it costs 25 euros per person for the session, and massages start at 60 euros. "
         "Children under 14 are not admitted."),
    ],
    "families_pets_accessibility.pdf": [
        ("Families and children",
         "Baby cots and high chairs are free on request while stock lasts. Children under 6 stay free when "
         "they use the existing beds. Each child receives a small welcome pack with a colouring book on "
         "arrival. The hotel does not run a kids club, but babysitting can be arranged for 15 euros per hour "
         "with 48 hours' notice. The restaurant offers a children's menu and the rooftop pool can be used by "
         "children with an adult."),
        ("Pets",
         "Dogs and cats up to 15 kg are welcome for a fee of 20 euros per stay. A pet bed and bowls are "
         "available on request at the front desk. Pets are not allowed in the restaurant or the wellness "
         "room, but they are tolerated on the roof terrace. Pets must be kept on a leash in all common "
         "areas. Guests are asked never to leave their pet alone in the room."),
        ("Accessibility",
         "The hotel has 3 accessible rooms on the ground floor, with a roll-in shower and grab bars. A lift "
         "serves all floors, including the rooftop pool. The restaurant is accessible to wheelchair users, "
         "while the wellness room can only be reached by stairs. Assistance dogs are welcome everywhere in "
         "the hotel at no charge, including in areas where pets are not allowed. Guests who need help are "
         "invited to tell the front desk before arrival."),
    ],
    "local_area_and_events.pdf": [
        ("Things to do nearby",
         "City bikes can be rented at the front desk for 12 euros per half day. The Old Harbour and its fish "
         "market are a 5-minute walk away; the market is held on Tuesday, Friday and Saturday mornings. The "
         "Cape Lanterna lighthouse trail starts 25 minutes on foot from the hotel. Boat trips around the bay "
         "leave from the Old Harbour pier, 6 minutes away. The Aqua Verde thermal baths are 20 minutes away "
         "by car and Valmora Golf Club is 15 minutes by car. In summer, the sandy beach of Cala Mar is 10 "
         "minutes away by tram."),
        ("Meetings and events",
         "The hotel has one meeting room for up to 24 people, equipped with a screen and video conferencing "
         "at no extra cost. A day package for meetings starts at 55 euros per person and includes the room, "
         "coffee breaks and lunch. The courtyard can be privatised for private events such as weddings, for "
         "up to 60 guests. An events coordinator helps organise everything from the first enquiry to the "
         "day itself."),
    ],
}


def _escape(text: str) -> bytes:
    """Encode for a PDF string in WinAnsi and escape the characters that have a meaning in PDF."""
    return text.encode("cp1252").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _page_stream(title: str, text: str, footer: str) -> bytes:
    lines = textwrap.wrap(text, width=92)
    body = b"BT /F2 11 Tf 56 745 Td 15 TL\n" + b"".join(b"(" + _escape(line) + b") Tj T*\n" for line in lines) + b"ET\n"
    return (
        b"BT /F1 18 Tf 56 780 Td (" + _escape(title) + b") Tj ET\n"
        + body
        + b"BT /F2 8 Tf 56 40 Td (" + _escape(footer) + b") Tj ET\n"
    )


def build_pdf(pages: list[tuple[str, str]]) -> bytes:
    """A minimal PDF: one A4 page per (title, text), Helvetica, with a footer on every page."""
    n = len(pages)
    page_ids = [5 + 2 * i for i in range(n)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [" + b" ".join(f"{i} 0 R".encode() for i in page_ids) + b"] /Count %d >>" % n,
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for i, (title, text) in enumerate(pages):
        footer = f"{HOTEL} - {ADDRESS} - Internal documentation - page {i + 1}"
        stream = _page_stream(title, text, footer)
        content_id = page_ids[i] + 1
        objects[page_ids[i]] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % content_id
        )
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += b"%d 0 obj\n" % obj_id + objects[obj_id] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for obj_id in sorted(objects):
        out += b"%010d 00000 n \n" % offsets[obj_id]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def write_documents(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, pages in DOCUMENTS.items():
        path = folder / name
        path.write_bytes(build_pdf(pages))
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "sample_hotel")
    for path in write_documents(parser.parse_args().out):
        print(f"wrote {path}")
